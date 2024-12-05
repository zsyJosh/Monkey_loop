import argparse
import copy
import json
import logging
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from os import path as osp
from random import sample
from typing import Any, Dict, List, Optional

import uuid
import chromadb
import dspy
import jsonlines
import numpy as np
import pandas as pd
import sentence_transformers
import tiktoken
import tqdm
from chromadb.config import Settings
from datasets import Dataset, DatasetDict
from dspy.datasets import DataLoader
from dspy.predict.avatar import Avatar, Tool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.utilities import (ArxivAPIWrapper, GoogleSerperAPIWrapper,
                                         WikipediaAPIWrapper)

# Configure warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from new_optimizer import AvatarOptimizerWithMetrics


OPENAI_API_KEY = ''
SERPER_API_KEY = ''

# add environment variables

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
HF_USR_NAME = 'shirwu'
TOOL_QA_ROOT = '/dfs/project/kgrlm/shirwu/msr_intern/home/t-yingxinwu/msr_intern/ToolQA-rebuttal'
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PERSIST_DIRECTORY = osp.join(TOOL_QA_ROOT, "data/chroma_db/scirex-v2")
CHROMA_COLLECTION_NAME = "all"
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_HTTP_PORT = "8000"
FILE_PATH = osp.join(TOOL_QA_ROOT, "data/external_corpus/scirex/Preprocessed_Scirex.jsonl")


logger = logging.getLogger(__name__)

# Disable all INFO logging
logging.getLogger().setLevel(logging.WARNING)

# Silence all loggers that might be chatty
loggers_to_silence = [
    "httpx",
    "httpcore",
    "openai",
    "arxiv",
    "dspy",
    "langchain",
    "langchain_community",
    "requests",
    "urllib3",
    "tiktoken",
    "asyncio",
    "faiss",
    "anthropic"
]

for logger_name in loggers_to_silence:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism warning


def RETRIEVE(query: str) -> str:
        """If you want to search for some paper information, you can use this tool and input a natural language query. For example, RETRIEVE(\'Which method achieves the highest PCK score?\') returns relevant paper paragraph and meta data."""
        return query_llm(query)

tools = [
    Tool(
        tool=StructuredTool.from_function(RETRIEVE),
        name="RETRIEVE",
        desc="If you want to search for some paper information, you can use this tool and input a natural language query. For example, RETRIEVE('Which method achieves the highest PCK score?') returns relevant paper paragraph and meta data."
    ),
    Tool(
        tool=GoogleSerperAPIWrapper(),
        name="WEB_SEARCH",
        desc="If you have a question, you can use this tool to search the web for the answer."
    ),
    Tool(
        tool=ArxivAPIWrapper(),
        name="ARXIV_SEARCH",
        desc="Pass the arxiv paper id to get the paper information.",
        input_type="Arxiv Paper ID",
    )
]

class ToolQASignature(dspy.Signature):
        """You will be given a question. Your task is to answer the question with a short response. 
        """
        
        question: str = dspy.InputField(
            prefix="Question:",
            desc="question to ask",
            format=lambda x: x.strip(),
        )
        answer: str = dspy.OutputField(
            prefix="Answer:",
            desc="answer to the question",
        )

class Evaluator(dspy.Signature):
    """Please act as an impartial judge to evaluate whether the answer is correct based on the ground truth answer"""
    
    question: str = dspy.InputField(
        prefix="Question:",
        desc="question to ask",
    )
    reference_answer: str = dspy.InputField(
        prefix="Ground Truth Answer:",
        desc="Ground truth answer to the question.",
    )
    answer: str = dspy.InputField(
        prefix="Answer:",
        desc="Answer to the question given by the model.",
    )
    rationale: str = dspy.OutputField(
        prefix="Rationale:",
        desc="Explanation of why the answer is correct or incorrect.",
    )
    is_correct: float = dspy.OutputField(
        prefix="Correct:",
        desc="Whether the answer is correct. Give 0 if incorrect, 1 if correct, (0, 1) if partially correct.",
    )


evaluator = dspy.TypedPredictor(Evaluator)


def metric(example, prediction, trace=None):  
    # We found sometimes the ground truth answers are incomplete or the answer
    # is part of the ground truth answer. Therefore, for better comparison, 
    # we use a continuous value for the correct score   
    acc = float(
        evaluator(
            question=example.question,
            answer=prediction.answer,
            reference_answer=example.answer
        ).is_correct
    ) 
    print(prediction.answer, '|', example.answer, '=>', acc)
    return acc


@dataclass
class APICallMetrics:
    timestamp: datetime
    tool_name: str
    tokens_in: int = 0
    tokens_out: int = 0
    execution_time: float = 0.0

@dataclass
class AvatarMetrics:
    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_execution_time: float = 0.0
    calls_by_tool: Dict[str, int] = field(default_factory=dict)
    api_call_history: List[APICallMetrics] = field(default_factory=list)
    
    def add_call(self, metrics: APICallMetrics):
        self.total_calls += 1
        self.total_tokens_in += metrics.tokens_in
        self.total_tokens_out += metrics.tokens_out
        self.total_execution_time += metrics.execution_time
        self.calls_by_tool[metrics.tool_name] = self.calls_by_tool.get(metrics.tool_name, 0) + 1
        self.api_call_history.append(metrics)
    
    def merge(self, other: 'AvatarMetrics'):
        """Merge another AvatarMetrics instance into this one"""
        self.total_calls += other.total_calls
        self.total_tokens_in += other.total_tokens_in
        self.total_tokens_out += other.total_tokens_out
        self.total_execution_time += other.total_execution_time
        for tool, count in other.calls_by_tool.items():
            self.calls_by_tool[tool] = self.calls_by_tool.get(tool, 0) + count
        self.api_call_history.extend(other.api_call_history)

    def estimate_cost(self, model_name: str = "gpt-4") -> float:
        pricing = {
            "gpt-4": {"input": 2.5, "output": 10.0},
        }
        if model_name not in pricing:
            raise ValueError(f"Unknown model: {model_name}")
        
        rates = pricing[model_name]
        input_cost = (self.total_tokens_in / 1000000) * rates["input"]
        output_cost = (self.total_tokens_out / 1000000) * rates["output"]
        return input_cost + output_cost

class AvatarWithMetrics(Avatar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = AvatarMetrics()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(str(text)))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return 0

    def _wrapped_tool_call(self, tool, input_text: str) -> str:
        start_time = time.time()
        tokens_in = self._count_tokens(input_text)
        
        try:
            result = tool.run(input_text)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            tokens_out = self._count_tokens(str(result))
            
            metrics = APICallMetrics(
                timestamp=datetime.now(),
                tool_name=tool.name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                execution_time=execution_time
            )
            self.metrics.add_call(metrics)
            
        return result

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = super().__call__(*args, **kwargs)
        total_time = time.time() - start_time
        
        metrics = APICallMetrics(
            timestamp=datetime.now(),
            tool_name="main_llm",
            tokens_in=self._count_tokens(str(args) + str(kwargs)),
            tokens_out=self._count_tokens(str(result)),
            execution_time=total_time
        )
        self.metrics.add_call(metrics)
        
        return result

def multi_thread_executor(test_set, signature, num_threads=60):
    total_score = 0
    total_examples = len(test_set)
    combined_metrics = AvatarMetrics()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for example in test_set:
            def process_with_metrics(example=example):
                try:
                    avatar = AvatarWithMetrics(signature, tools=tools, verbose=False, max_iters=10)
                    prediction = avatar(**example.inputs().toDict())
                    return metric(example, prediction), avatar.metrics
                except Exception as e:
                    print(e)
                    return 0, AvatarMetrics()

            futures.append(executor.submit(process_with_metrics))

        for future in tqdm.tqdm(futures, total=total_examples, desc="Processing examples"):
            score, metrics = future.result()
            total_score += score
            # Only combine token counts and call counts, not execution times
            combined_metrics.total_calls += metrics.total_calls
            combined_metrics.total_tokens_in += metrics.total_tokens_in
            combined_metrics.total_tokens_out += metrics.total_tokens_out
            for tool, count in metrics.calls_by_tool.items():
                combined_metrics.calls_by_tool[tool] = combined_metrics.calls_by_tool.get(tool, 0) + count
            combined_metrics.api_call_history.extend(metrics.api_call_history)
    
    total_execution_time = time.time() - start_time
    combined_metrics.total_execution_time = total_execution_time

    avg_metric = total_score / total_examples
    return avg_metric, combined_metrics

def single_thread_executor(test_set, signature):
    total_score = 0
    total_examples = len(test_set)
    combined_metrics = AvatarMetrics()

    for example in tqdm.tqdm(test_set, desc="Processing examples"):
        try:
            avatar = AvatarWithMetrics(signature, tools=tools, verbose=False, max_iters=10)
            prediction = avatar(**example.inputs().toDict())
            score = metric(example, prediction)
            total_score += score
            # Combine metrics from this run
            for call in avatar.metrics.api_call_history:
                combined_metrics.add_call(call)
        except Exception as e:
            print(e)

    avg_metric = total_score / total_examples
    return avg_metric, combined_metrics

def format_metrics_report(metrics: AvatarMetrics, model_name: str = "gpt-4") -> str:
    cost = metrics.estimate_cost(model_name)
    
    report = f"""
Avatar Execution Metrics Report
==============================
Execution Time: {metrics.total_execution_time:.2f} seconds
Total API Calls: {metrics.total_calls}
Total Tokens: {metrics.total_tokens_in + metrics.total_tokens_out:,} ({metrics.total_tokens_in:,} in, {metrics.total_tokens_out:,} out)
Estimated Cost: ${cost:.4f}

Average Time per Call: {metrics.total_execution_time / metrics.total_calls:.2f} seconds

Tool Usage Breakdown:
-------------------
"""
    for tool, count in sorted(metrics.calls_by_tool.items()):
        report += f"{tool}: {count} calls\n"

    return report


class ExperimentRunner:
    def __init__(self, toolqa_train, toolqa_test, actor_agent, experiment_config=None):
        self.toolqa_train = toolqa_train
        self.toolqa_test = toolqa_test
        self.actor_agent = actor_agent
        self.experiment_config = experiment_config or {}
        self.results = {
            "strategies": {},
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "config": experiment_config
        }

    def run_one_shot(self) -> Dict[str, Any]:
        """Run one-shot generation strategy"""
        try:
            score, metrics = multi_thread_executor(self.toolqa_test, ToolQASignature)
            return {
                "performance": score,
                "execution_time": metrics.total_execution_time,
                "cost": metrics.estimate_cost(),
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_batch_sampling(self, batch_num: int = 2) -> Dict[str, Any]:
        """Run batch sampling strategy"""
        try:
            batch_sampling_monkey = AvatarOptimizerWithMetrics(
                metric=metric,
                max_iters=0,
                max_negative_inputs=10,
                max_positive_inputs=10,
                lower_bound=0.5,
                upper_bound=0.5
            )
            result = batch_sampling_monkey.compile(
                student=self.actor_agent,
                trainset=self.toolqa_train
            )
            score, batch_metrics = batch_sampling_monkey.thread_safe_evaluator_batch(
                self.toolqa_test, 
                result['agent'], 
                batch_num
            )
            return {
                "performance": score,
                "execution_time": batch_metrics['total_execution_time'],
                "cost": batch_metrics['total_cost'],
                "best_batch_score": batch_metrics['final_score'],
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_iterative_evolution(self, max_iters: int = 1) -> Dict[str, Any]:
        """Run iterative evolution strategy"""
        try:
            iterative_monkey = AvatarOptimizerWithMetrics(
                metric=metric,
                max_iters=max_iters,
                max_negative_inputs=10,
                max_positive_inputs=10,
                lower_bound=0.5,
                upper_bound=0.5
            )
            result = iterative_monkey.compile(
                student=self.actor_agent,
                trainset=self.toolqa_train
            )
            optimized_actor = result["agent"]
            optimization_metrics = result["metrics"]
            
            score, eval_metrics = iterative_monkey.thread_safe_evaluator(
                self.toolqa_test, 
                optimized_actor
            )
            
            return {
                "performance": score,
                "execution_time": eval_metrics['execution_time'],
                "cost": eval_metrics['total_cost'],
                "optimization_cost": optimization_metrics['total_cost'],
                "optimization_time": optimization_metrics['total_execution_time'],
                "iteration_details": optimization_metrics['iteration_details'],
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_mixed_strategy(self, max_iters: int = 1, batch_num: int = 2) -> Dict[str, Any]:
        """Run mixed strategy (iterative evolution + batch evaluation)"""
        try:
            # First run iterative evolution to get optimized agent
            iterative_monkey = AvatarOptimizerWithMetrics(
                metric=metric,
                max_iters=max_iters,
                max_negative_inputs=10,
                max_positive_inputs=10,
                lower_bound=0.5,
                upper_bound=0.5
            )
            result = iterative_monkey.compile(
                student=self.actor_agent,
                trainset=self.toolqa_train
            )
            optimized_actor = result["agent"]
            optimization_metrics = result["metrics"]
            
            # Then do batch evaluation
            score, batch_metrics = iterative_monkey.thread_safe_evaluator_batch(
                self.toolqa_test, 
                optimized_actor,
                batch_num
            )
            
            return {
                "performance": score,
                "execution_time": batch_metrics['total_execution_time'],
                "cost": batch_metrics['total_cost'],
                "optimization_cost": optimization_metrics['total_cost'],
                "optimization_time": optimization_metrics['total_execution_time'],
                "best_batch_score": batch_metrics['final_score'],
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_experiments(self, strategies: Optional[list] = None) -> Dict[str, Any]:
        """Run specified strategies and collect results"""
        if strategies is None:
            strategies = ['one_shot', 'batch_sampling', 'iterative_evolution', 'mixed']
        
        max_iters = self.experiment_config.get('max_iters', 1)
        batch_size = self.experiment_config.get('batch_size', 2)
        
        for strategy in strategies:
            if strategy == 'one_shot':
                self.results['strategies']['one_shot'] = self.run_one_shot()
            elif strategy == 'batch_sampling':
                self.results['strategies']['batch_sampling'] = self.run_batch_sampling(batch_size)
            elif strategy == 'iterative_evolution':
                self.results['strategies']['iterative_evolution'] = self.run_iterative_evolution(max_iters)
            elif strategy == 'mixed':
                self.results['strategies']['mixed'] = self.run_mixed_strategy(max_iters, batch_size)
        
        return self.results

    def save_results(self, output_path: str):
        """Save results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

def sentence_embedding(model, texts):
    embeddings = model.encode(texts)
    return embeddings

def create_chroma_db(chroma_server_host, chroma_server_http_port, collection_name):
    chroma_client = chromadb.Client(Settings(
        chroma_api_impl="rest",
        chroma_server_host=chroma_server_host,
        chroma_server_http_port=chroma_server_http_port,
    ))
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def create_chroma_db_local(persist_directory, collection_name):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def insert_to_db(texts, model_name, cuda_idx, db):
    # use cpu
    model = sentence_transformers.SentenceTransformer(model_name, device='cpu')
    # model = sentence_transformers.SentenceTransformer(model_name, device=f"cuda:{cuda_idx}")

    batch_embeddings = []
    batch_texts = []
    start_time = time.time()
    print(f"Total Articles to process: {len(texts)}, Current Thread: {cuda_idx}.")
    for i, text in enumerate(texts):
        # 2. generate embedding
        embeddings = sentence_embedding(model, text).tolist()

        batch_embeddings.append(embeddings)
        batch_texts.append(text)
        # 3. add to vectorstore per 500 articles or last article
        if i % 100 == 0 or i == len(texts)-1:
            batch_ids = [str(uuid.uuid1()) for _ in batch_texts]
            db.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids = batch_ids
            )
            batch_embeddings = []
            batch_texts = []
            print(f"Completed Processing article count: {i}, Current Thread: {cuda_idx}, Time took: {time.time() - start_time}.")
    print(f"Thread {cuda_idx} Completed. Total time took for thread: {time.time() - start_time}.")


# Multi-processing
def query_llm(query, is_local=True, start=None, end=None):
    cuda_idxes = [0]
    number_of_processes = len(cuda_idxes)
    input_texts = []
    db = create_chroma_db_local(CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME)
    with open(FILE_PATH, 'r') as f:
        for item in jsonlines.Reader(f):
            input_texts.append(item["content"])
    # input_texts = np.array_split(input_texts, number_of_processes)

    args = ((input_texts[i], EMBED_MODEL_NAME, cuda_idxes[i], is_local) for i in range(number_of_processes))

    # if there is no file under the directory "/localscratch/yzhuang43/ra-llm/retrieval_benchmark/data/chroma_db/agenda", insert the data into the db
    # You should run insert_to_db the first time!
    if len(os.listdir(CHROMA_PERSIST_DIRECTORY)) == 0:
        insert_to_db(input_texts, model_name=EMBED_MODEL_NAME, cuda_idx=0, db=db)

    input_paths = np.array_split(input_texts, number_of_processes)
    with ProcessPoolExecutor(number_of_processes) as executor:
        executor.map(insert_to_db, args)
    # use cpu
    model = sentence_transformers.SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
    # model = sentence_transformers.SentenceTransformer(EMBED_MODEL_NAME, device=f"cuda:0")
    query_embedding = sentence_embedding(model, query).tolist()
    results = db.query(query_embeddings=query_embedding, n_results=3)
    retrieval_content = [result for result in results['documents'][0]]
    # print(retrieval_content)
    retrieval_content = '\n'.join(retrieval_content)
    return retrieval_content


def main():
    parser = argparse.ArgumentParser(description='Run evaluation experiments')
    parser.add_argument('--strategies', nargs='+', 
                      choices=['one_shot', 'batch_sampling', 'iterative_evolution', 'mixed'],
                      help='Strategies to run')
    parser.add_argument('--output', type=str, default='results.json',
                      help='Output JSON file path')
    parser.add_argument('--batch-size', type=int, default=2,
                      help='Batch size for batch sampling and mixed strategies')
    parser.add_argument('--max-iters', type=int, default=1,
                      help='Maximum iterations for iterative evolution')
    parser.add_argument('--experiment-name', type=str, default='',
                      help='Name of the experiment for logging purposes')
    
    args = parser.parse_args()

    # initialize settings
    level = 'hard'
    dataset = 'scirex'
    dataset_dir = f'{dataset}-{level}.jsonl'
    hf_dataset_name = f'toolqa_{dataset}_{level}'
    df = pd.read_json(dataset_dir, lines=True)
    df.head()
    df['answer'] = df['answer'].apply(lambda x: str(x))
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({'train': dataset})
    dspy.settings.configure(
        lm=dspy.OpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            max_tokens=4000,
            temperature=0
        )
    )

    dl = DataLoader()
    tool_qa = dl.from_huggingface(
        f'{HF_USR_NAME}/' + hf_dataset_name,
        split="train",
        input_keys=("question", "answer"),
    )
    random.seed(42)

    train_idx = random.sample(range(len(tool_qa)), 40)
    remaining_idx = list(set(range(len(tool_qa))) - set(train_idx))
    test_idx = random.sample(remaining_idx, 60)

    toolqa_train = [
        dspy.Example(question=example.question, answer=example.answer).with_inputs("question", "paper_id")
        for example in [tool_qa[i] for i in train_idx]
    ]
    toolqa_test = [
        dspy.Example(question=example.question, answer=example.answer).with_inputs("question", "paper_id")
        for example in [tool_qa[i] for i in test_idx]
    ]


    actor_agent = Avatar(
        tools=tools,
        signature=ToolQASignature,
        verbose=False,
        max_iters=10
    )

    experiment_config = {
        'max_iters': args.max_iters,
        'batch_size': args.batch_size,
        'experiment_name': args.experiment_name
    }

    
    # Create experiment runner
    runner = ExperimentRunner(toolqa_train, toolqa_test, actor_agent, experiment_config)
    
    # Run experiments
    results = runner.run_experiments(args.strategies)
    
    # Save results
    runner.save_results(args.output)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()