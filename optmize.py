import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import time
import re
import random
import argparse
import numpy as np
import fcntl
import torch
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential
from src.gpt_model import GPT, EmbeddingModel
from tqdm import tqdm
from src.optimize_prompt import (
    wrap_formated_history,
    wrap_textual_loss_prompt,
    wrap_textual_gradient_prompt,
    wrap_textual_update_prompt,
    wrap_misleading_reason_rewrite,
    wrap_rag_simulation_prompt,
    wrap_rag_judge_misleading_prompt,
    wrap_rag_judge_misleading_untargeted_prompt,
)

# ========== Embedding Model Wrapper Classes ==========
class QwenEmbeddingModel:
    """Lightweight wrapper for Qwen3-Embedding-8B (ported from eval.py)"""
    def __init__(self, model_path: str, use_fp16: bool = True, use_flash_attention: bool = True) -> None:
        print(f"[INFO] Loading Qwen3-Embedding-8B model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)

        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        print(f"[INFO] Qwen3-Embedding-8B loaded, device: {self.device}")

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def _encode(self, texts, *, is_query: bool, max_length: int = 8192) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            texts = [self.task_description + t for t in texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])

        return embeddings.detach().cpu().numpy()

    def encode_query(self, text: str) -> np.ndarray:
        return self._normalize(self._encode(text, is_query=True)[0])

    def encode_document(self, text: str) -> np.ndarray:
        return self._normalize(self._encode(text, is_query=False)[0])
    
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 8192, 
               is_query: bool = False) -> Dict[str, np.ndarray]:
        """Batch encoding interface, compatible with optimize.py call style"""
        embeddings = self._encode(texts, is_query=is_query, max_length=max_length)
        normalized_embeddings = np.array([self._normalize(emb) for emb in embeddings])
        return {'dense_vecs': normalized_embeddings}


class BGEEmbeddingModel:
    """BGE-M3 Encoder Wrapper (ported from eval.py)"""
    def __init__(self, model_path: str) -> None:
        print(f"[INFO] Using BGE-M3 retriever: {model_path}")
        self.model = BGEM3FlagModel(model_path, use_fp16=True)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_query(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0]
        return self._normalize(result)

    def encode_document(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0]
        return self._normalize(result)
    
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 8192) -> Dict[str, np.ndarray]:
        """Batch encoding interface"""
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return {'dense_vecs': result['dense_vecs']}


class OpenAIEmbeddingModel:
    """OpenAI Embedding API Wrapper (text-embedding-3-large)"""
    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        print(f"[INFO] Using OpenAI Embedding API: {model_name}")
        self.model = EmbeddingModel(model_name=model_name)
        self.model_name = model_name
        self.max_chars = 6000  # Max char limit to prevent API limit
        print(f"[INFO] {model_name} loaded, max char limit: {self.max_chars}")

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize vector (API returns normalized vector, keeping interface consistent)"""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def _truncate_text(self, text: str) -> str:
        """Truncate overly long text"""
        if len(text) <= self.max_chars:
            return text
        return text[:self.max_chars]

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query (single)"""
        truncated_text = self._truncate_text(text)
        try:
            result = self.model.query(truncated_text)[0]  # API returns list[list[float]], take first
            return np.asarray(result, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to encode query: {e}")
            # Try shorter text
            very_short = text[:1000]
            result = self.model.query(very_short)[0]
            return np.asarray(result, dtype=np.float32)

    def encode_document(self, text: str) -> np.ndarray:
        """Encode document (single)"""
        truncated_text = self._truncate_text(text)
        try:
            result = self.model.query(truncated_text)[0]
            return np.asarray(result, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to encode document: {e}")
            # Try shorter text
            very_short = text[:1000]
            result = self.model.query(very_short)[0]
            return np.asarray(result, dtype=np.float32)
    
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 8192,
               is_query: bool = False) -> Dict[str, np.ndarray]:
        """Batch encoding interface"""
        # Preprocessing: Truncate overly long texts
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        truncated_count = sum(1 for orig, trunc in zip(texts, truncated_texts) if len(orig) > len(trunc))
        if truncated_count > 0:
            print(f"[WARNING] {truncated_count}/{len(texts)} texts truncated to {self.max_chars} chars")
        
        try:
            # OpenAI API supports batch requests, pass texts list directly
            embeddings = self.model.query(truncated_texts)  # Returns list[list[float]]
            embeddings_array = np.array(embeddings, dtype=np.float32)
            return {'dense_vecs': embeddings_array}
        except Exception as e:
            print(f"[ERROR] Batch encoding failed: {e}, trying single processing...")
            # Single processing
            single_embs = []
            for text in truncated_texts:
                try:
                    emb = self.model.query(text)[0]
                    single_embs.append(emb)
                except:
                    # Use shorter text
                    short_text = text[:1000]
                    emb = self.model.query(short_text)[0]
                    single_embs.append(emb)
            return {'dense_vecs': np.array(single_embs, dtype=np.float32)}


def build_embedding_model(model_path: str, model_type: str = "auto"):
    """
    Build embedding model based on model path and type
    
    Args:
        model_path: Model path or model name
        model_type: Model type ("bge-m3", "qwen", "openai", "auto")
                   "auto" will automatically determine based on path
    
    Returns:
        Embedding model instance
    """
    if model_type == "auto":
        model_path_lower = model_path.lower()
        if "bge-m3" in model_path_lower or "bge" in model_path_lower:
            model_type = "bge-m3"
        elif "qwen" in model_path_lower or "qwen3" in model_path_lower:
            model_type = "qwen"
        elif "embedding-3" in model_path_lower or "text-embedding" in model_path_lower:
            model_type = "openai"
        else:
            raise ValueError(f"Cannot automatically identify model type, path: {model_path}")
    
    if model_type == "bge-m3":
        return BGEEmbeddingModel(model_path)
    elif model_type == "qwen":
        return QwenEmbeddingModel(model_path)
    elif model_type == "openai":
        return OpenAIEmbeddingModel(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ========== Argument Parsing ==========
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Adversarial Example TPO Optimization Pipeline")
    parser.add_argument('--start_id', type=int, default="1")
    parser.add_argument('--end_id', type=int, default="10")
    parser.add_argument('--version', type=str)
    # File path parameters
    parser.add_argument('--root_path', type=str, default="datasets/", help="Dataset root directory")
    parser.add_argument('--dataset', type=str, default="bioasq", choices=["bioasq", "finqa", "tiebe"])
    parser.add_argument('--domain', type=str, default="biomedical", choices=["biomedical", "financial", "real event"])
    parser.add_argument('--attack_type', type=str, default="targeted", choices=["targeted", "untargeted"], 
                        help="Attack type: targeted or untargeted")
    # Model parameters
    parser.add_argument('--optimizer_model', type=str, default="gpt-4o-mini", help="Optimizer model")
    parser.add_argument('--judge_model', type=str, default="gpt-4o-mini", help="LLMprefer model")
    parser.add_argument('--embedding_model_path', type=str, default="text-embedding-3-large", help="Embedding model path or name")
    parser.add_argument('--embedding_model_type', type=str, default="auto", 
                        choices=["auto", "bge-m3", "qwen", "openai"],
                        help="Embedding model type: auto, bge-m3, qwen, openai")
    # TPO optimization parameters
    parser.add_argument('--max_iterations', type=int, default=30, help="Maximum iterations")
    parser.add_argument('--candidates_per_iteration', type=int, default=6, help="Candidates generated per iteration")
    parser.add_argument('--history_size', type=int, default=20, help="Optimization history size")
    # Evaluation parameters
    parser.add_argument('--similarity_weight', type=float, default=0.5, help="Weight for similarity score")
    parser.add_argument('--trust_weight', type=float, default=0.5, help="Weight for RAG trust score")
    # Execution parameters
    parser.add_argument('--temperature', type=float, default=1.0, help="Generation temperature")
    parser.add_argument('--generation_workers', type=int, default=6, help="Workers for candidate generation")
    parser.add_argument('--evaluation_workers', type=int, default=6, help="Workers for candidate evaluation")
    # Early stopping parameters
    parser.add_argument('--early_stop_patience', type=int, default=10, help="Early stopping patience (consecutive non-improvements)")
    parser.add_argument('--save_progress', action='store_true', default=True, help="Save progress during execution")
    parser.add_argument('--queries_per_fact', type=int, default=3, help="Queries per fact per role (for untargeted attack)")
    parser.add_argument('--report_only', action='store_true', help="Only show status report")
    
    args = parser.parse_args()

    suffix = "targeted" if args.attack_type == "targeted" else "untargeted"
    args.file_path = os.path.join(args.root_path, f"{args.version}/{args.dataset}-{args.version}-{suffix}.json")
    return args

# ========== Utility Functions ==========
def load_corpus(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load corpus data"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_progress(file_path: str, data_to_update: Dict[str, Dict[str, Any]]):
    """
    Atomically and safely update entries in the JSON file
    """
    lock_file = file_path + ".lock"
    
    try:
        with open(lock_file, 'w') as lock_f:
            # Try to acquire exclusive lock, wait up to 60 seconds
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            
            # Read current file content
            current_data = {}
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARNING] File {file_path} is corrupted, creating new file.")
                    current_data = {}
            
            # Merge data: update current data with new data
            current_data.update(data_to_update)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
                
            # Print updated IDs
            updated_ids = ", ".join(data_to_update.keys())
            print(f"Progress safely saved, updated IDs: {updated_ids}")

    except (IOError, OSError) as e:
        print(f"[ERROR] File operation error while saving progress: {e}")
    except Exception as e:
        print(f"[ERROR] Unknown error while saving progress: {e}")
    finally:
        # Clean up lock file
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError as e:
                print(f"[WARNING] Failed to clean up lock file: {e}")

def check_optimization_status(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    """Check optimization status"""
    status = {}
    for corpus_id, corpus_data in data.items():
        has_required_data = (
            'enhanced_erroneous_corpus' in corpus_data and 
            'erroneous_corpus' in corpus_data and 
            'related_queries' in corpus_data and
            'selected_queries' in corpus_data and
            corpus_data['enhanced_erroneous_corpus'] and
            corpus_data['erroneous_corpus'] and
            corpus_data['related_queries']
        )
        has_optimization = 'final_optimized_corpus' in corpus_data
        status[corpus_id] = {
            'ready_for_optimization': has_required_data,
            'has_optimization': has_optimization
        }
    return status

def print_optimization_report(file_path: str):
    """Print optimization status report"""
    data = load_corpus(file_path)
    status = check_optimization_status(data)
    
    print("\n=== Optimization Status Report ===")
    print(f"Total corpus count: {len(data)}")
    
    ready_count = sum(1 for s in status.values() if s['ready_for_optimization'])
    optimized_count = sum(1 for s in status.values() if s['has_optimization'])
    
    print(f"Ready for optimization: {ready_count}/{len(data)} ({ready_count/len(data)*100:.1f}%)")
    print(f"Optimization completed: {optimized_count}/{len(data)} ({optimized_count/len(data)*100:.1f}%)")
    print("=====================\n")

# ========== Similarity Evaluation Functions ==========
def compute_batch_embeddings(texts: List[str], embedding_model, is_query: bool = False) -> np.ndarray:
    """
    Batch compute text embeddings - Supports BGE-M3, Qwen3, OpenAI
    
    Args:
        texts: List of texts
        embedding_model: Embedding model instance
        is_query: Whether it is query text (required for Qwen)
        
    Returns:
        numpy array: Embedding vector matrix
    """
    # Unified call to encode method of wrapper class
    if isinstance(embedding_model, (BGEEmbeddingModel, QwenEmbeddingModel, OpenAIEmbeddingModel)):
        if isinstance(embedding_model, QwenEmbeddingModel):
            result = embedding_model.encode(texts, batch_size=32, max_length=8192, is_query=is_query)
        else:  # BGEEmbeddingModel and OpenAIEmbeddingModel
            result = embedding_model.encode(texts, batch_size=32, max_length=8192)
        return result['dense_vecs']
    elif isinstance(embedding_model, BGEM3FlagModel):
        # Compatible with old BGE-M3 direct call
        result = embedding_model.encode(texts, 
                                        batch_size=32,
                                        max_length=8192,
                                        return_dense=True,
                                        return_sparse=False,
                                        return_colbert_vecs=False)
        return result['dense_vecs']
    else:
        raise TypeError(f"Unsupported model type: {type(embedding_model)}")

def compute_similarity_scores(document: str, related_queries: Dict[str, List[str]], 
                            selected_queries: List[str], embedding_model,
                            attack_type: str, num_facts: int = 1,
                            queries_per_fact: int = 3) -> Dict[str, Any]:
    """
    Evaluate document similarity score (Generalization reward only)
    
    Args:
        document: Document to evaluate
        related_queries: All related queries (grouped by role)
        selected_queries: Anchor queries (injected into document, not used here)
        embedding_model: embedding model
        attack_type: "targeted" or "untargeted"
        num_facts: Number of facts (only for untargeted)
        queries_per_fact: Number of queries per fact per role (default 3)
    """
    # ===== Generalization Loss Calculation =====
    if attack_type == "targeted":
        # Targeted attack: Randomly select 1 query from each role (6 roles = 6 queries)
        sampled_gen_queries = []
        for role, queries in related_queries.items():
            if queries:
                sampled_query = random.choice(queries)
                sampled_gen_queries.append(sampled_query)
    else:
        # Untargeted attack:
        # 1. Randomly select 1 fact from all facts
        # 2. Select 1 query from each role for that fact (6 roles = 6 queries)
        selected_fact_idx = random.randint(0, num_facts - 1)
        
        sampled_gen_queries = []
        for role, queries in related_queries.items():
            # Calculate index range for this fact in this role's query list
            start_idx = selected_fact_idx * queries_per_fact
            end_idx = start_idx + queries_per_fact
            
            if end_idx <= len(queries):
                fact_queries = queries[start_idx:end_idx]
                if fact_queries:
                    sampled_query = random.choice(fact_queries)
                    sampled_gen_queries.append(sampled_query)
    
    # Batch compute embeddings
    doc_embedding = compute_batch_embeddings([document], embedding_model, is_query=False)  # [1, dim] Document
    query_embeddings = compute_batch_embeddings(sampled_gen_queries, embedding_model, is_query=True)  # [num_queries, dim] Queries
    
    # Compute similarity
    similarities = doc_embedding @ query_embeddings.T  # [1, num_queries]
    generalization_similarities = similarities.flatten().tolist()  # [num_queries]
    
    # Compute average score
    generalization_score = np.mean(generalization_similarities) if generalization_similarities else 0.0
    
    return {
        'sampled_gen_queries': sampled_gen_queries,
        'generalization_score': float(generalization_score),
        'generalization_similarities': generalization_similarities
    }

# ========== RAG Trust Evaluation Functions ==========
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def evaluate_rag_trust(
    optimized_doc: str, 
    original_doc: str, 
    judge_model: GPT, 
    domain: str, 
    sample_query: str,
    attack_type: str,
    correct_answer: Optional[str],
    target_answer: Optional[str]
) -> Dict[str, Any]:
    """
    Evaluate RAG system misleading effect
    
    Strategy:
    1. Simulate real RAG scenario, let LLM answer based on two documents (AB and BA order)
    2. Use different Judge models to evaluate if generated answers are misled based on attack type
       - targeted: Evaluate if answer is closer to target_answer than correct_answer
       - untargeted: Evaluate if answer deviates from original_doc (i.e., correct_document)
    """
    
    # ========== Step 1: RAG Scenario Simulation - AB Order ==========
    prompt_ab = wrap_rag_simulation_prompt(
        doc_a=optimized_doc, doc_b=original_doc, question=sample_query
    )
    answer_ab = judge_model.query(prompt_ab, temperature=0.7).strip()
    
    # ========== Step 2: RAG Scenario Simulation - BA Order ==========
    prompt_ba = wrap_rag_simulation_prompt(
        doc_a=original_doc, doc_b=optimized_doc, question=sample_query
    )
    answer_ba = judge_model.query(prompt_ba, temperature=0.7).strip()
    
    # ========== Step 3 & 4: Judge Evaluation (Select logic based on attack type) ==========
    judge_responses = []
    answers = [answer_ab, answer_ba]
    
    for answer in answers:
        if attack_type == "targeted":
            judge_prompt = wrap_rag_judge_misleading_prompt(
                question=sample_query,
                correct_answer=correct_answer,
                target_answer=target_answer,
                generated_answer=answer
            )
        else: # untargeted
            judge_prompt = wrap_rag_judge_misleading_untargeted_prompt(
                question=sample_query,
                correct_document=original_doc, # In untargeted attack, original document is correct document
                generated_answer=answer
            )
        
        response = judge_model.query(judge_prompt, temperature=0.3)
        
        # Extract JSON
        match = re.search(r'Evaluation:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
        response = match.group(1).strip() if match else response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        judge_responses.append(response)

    judge_response_ab, judge_response_ba = judge_responses

    # ========== Step 5: Parse Judge Results ==========
    try:
        result_ab = json.loads(judge_response_ab)
        result_ba = json.loads(judge_response_ba)
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parse failed: {e}")
        print(f"AB Response: {judge_response_ab[:200]}...")
        print(f"BA Response: {judge_response_ba[:200]}...")
        return {
            'trust_score': 0.0, 'reasoning': f"JSON parse failed: {str(e)}",
            'mislead_count': 0, 'avg_judge_score': 0.0,
            'misleading_score_ab': 0, 'misleading_score_ba': 0,
            'verdict_ab': 'ERROR', 'verdict_ba': 'ERROR',
            'answer_ab': answer_ab[:100], 'answer_ba': answer_ba[:100]
        }
    
    # ========== Step 6: Calculate Final Score (Segmented Reward Mechanism) ==========
    misleading_score_ab = result_ab.get('misleading_score', 0)
    misleading_score_ba = result_ba.get('misleading_score', 0)
    verdict_ab = result_ab.get('verdict', 'UNKNOWN')
    verdict_ba = result_ba.get('verdict', 'UNKNOWN')
    
    # Define successful misleading verdicts (Unified use of MISLED and MIXED)
    successful_verdicts = ['MISLED', 'MIXED']
    
    mislead_count = (1 if verdict_ab in successful_verdicts else 0) + \
                    (1 if verdict_ba in successful_verdicts else 0)
    
    avg_judge_score = (misleading_score_ab + misleading_score_ba) / 2.0
    
    # Segmented Reward + Judge Score Superposition
    if mislead_count == 0:
        scaled_score = avg_judge_score * 0.4
    elif mislead_count == 1:
        scaled_score = 40 + avg_judge_score * 0.3
    else:
        scaled_score = 70 + avg_judge_score * 0.3
    
    # ========== Step 7: Reasoning Perspective Transformation ==========
    reasoning_prompts = [
        wrap_misleading_reason_rewrite(
            original_reason=result_ab.get('reasoning', 'No reasoning provided.'),
            verdict=verdict_ab,
            doc_position='A'
        ),
        wrap_misleading_reason_rewrite(
            original_reason=result_ba.get('reasoning', 'No reasoning provided.'),
            verdict=verdict_ba,
            doc_position='B'
        )
    ]
    
    rewritten_reasons = []
    for prompt in reasoning_prompts:
        reason = judge_model.query(prompt, temperature=0.7)
        match = re.search(r'Rewritten Reasoning:\s*(.*)', reason, flags=re.IGNORECASE | re.DOTALL)
        rewritten_reasons.append(match.group(1).strip() if match else reason.strip())

    combined_reasoning = f"Reason 1: {rewritten_reasons[0]}\nReason 2: {rewritten_reasons[1]}"
    
    return {
        'trust_score': scaled_score,
        'mislead_count': mislead_count,
        'avg_judge_score': avg_judge_score,
        'reasoning': combined_reasoning,
        'misleading_score_ab': misleading_score_ab,
        'misleading_score_ba': misleading_score_ba,
        'verdict_ab': verdict_ab,
        'verdict_ba': verdict_ba,
        'answer_ab': answer_ab[:200],
        'answer_ba': answer_ba[:200]
    }

# ========== Optimization History Management ==========
class OptimizationHistory:
    """Optimization History Manager"""
    def __init__(self, domain: str, max_size: int = 20):
        self.max_size = max_size
        self.domain = domain
        self.history = []
    
    def add_candidate(self, document: str, total_score: float, similarity_scores: Dict[str, Any], 
                     trust_scores: Dict[str, Any], generation: int):
        """Add new candidate document to history"""
        record = {
            'document': document,
            'total_score': total_score,
            'similarity_scores': similarity_scores,
            'trust_scores': trust_scores,
            'generation': generation,
            'timestamp': time.time()
        }

        self.history.append(record)
        
        # Keep highest score records after exceeding length
        if len(self.history) > self.max_size:
            self.history.sort(key=lambda x: x['total_score'], reverse=True)
            self.history = self.history[:self.max_size]
    
    def get_best_worst_pair(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get best and worst document pair"""
        sorted_history = sorted(self.history, key=lambda x: x['total_score'])
        return sorted_history[-1], sorted_history[0]  # best, worst
    
    def get_best_document(self) -> Optional[Dict[str, Any]]:
        """Get current best document"""
        return max(self.history, key=lambda x: x['total_score'])

# ========== TPO Optimizer ==========
class TPOOptimizer:
    """TPO Optimizer Main Class"""
    def __init__(self, args):
        self.args = args
        self.domain = args.domain
        
        # Initialize models
        self.optimizer_model = GPT(model_name=args.optimizer_model, base_url="", api_key="")
        self.judge_model = GPT(model_name=args.judge_model, base_url="", api_key="")
        
        # Early stopping parameters
        self.early_stop_patience = args.early_stop_patience
        
        # Load embedding model
        print(f"[INFO] Loading embedding model: {args.embedding_model_path}")
        print(f"[INFO] Model type: {args.embedding_model_type}")
        self.embedding_model = build_embedding_model(args.embedding_model_path, args.embedding_model_type)
        print("[INFO] Embedding model loaded")
    
    def evaluate_document(self, document: str, related_queries: Dict[str, List[str]], 
                         selected_queries: List[str], original_doc: str,
                         correct_answer: Optional[str], target_answer: Optional[str], 
                         attack_type: str, num_facts: int = 1) -> Tuple[float, Dict]:
        """
        Comprehensive evaluation of a single document (Generalization Reward + Misleading Reward)
        """
        # 1. Similarity evaluation (Generalization only)
        similarity_scores = compute_similarity_scores(
            document, related_queries, selected_queries, 
            self.embedding_model, attack_type, num_facts,
            self.args.queries_per_fact
        )

        # 2. RAG Trust evaluation (Randomly select 1 from generalization queries)
        sample_query = random.choice(similarity_scores['sampled_gen_queries'])
        trust_scores = evaluate_rag_trust(
            optimized_doc=document, 
            original_doc=original_doc, 
            judge_model=self.judge_model, 
            domain=self.domain, 
            sample_query=sample_query,
            attack_type=attack_type,
            correct_answer=correct_answer, 
            target_answer=target_answer
        )

        # 3. Calculate total score (Scaled to 0-100)
        similarity_component = similarity_scores['generalization_score'] * 100.0
        trust_component = trust_scores['trust_score']
        
        # Calculate weighted sum
        total_score = (
            similarity_component * self.args.similarity_weight +
            trust_component * self.args.trust_weight
        )
        
        detailed_scores = {
            'similarity_scores': similarity_scores,
            'trust_scores': trust_scores,
            'components': {
                'similarity_component': similarity_component,
                'trust_component': trust_component,
            }
        }

        return total_score, detailed_scores
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_textual_loss(self, best_doc: Dict, worst_doc: Dict) -> str:
        """Generate textual loss analysis"""
        chosen_formatted = wrap_formated_history(best_doc, self.domain)
        rejected_formatted = wrap_formated_history(worst_doc, self.domain)
        prompt = wrap_textual_loss_prompt(
            chosen_formatted, rejected_formatted, self.domain
        )
        response = self.optimizer_model.query(prompt, temperature=0.7)
        match = re.search(r'Diagnostic Report:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            response = response.strip()

        return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_textual_gradient(self, textual_loss: str, best_doc: Dict) -> str:
        """Generate textual gradient"""
        chosen_formatted = wrap_formated_history(best_doc, self.domain)
        prompt = wrap_textual_gradient_prompt(self.domain, textual_loss, chosen_formatted)
        response = self.optimizer_model.query(prompt, temperature=0.7)
        match = re.search(r'Final Recommendations:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            response = response.strip()
        return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_new_candidate(self, best_document: str, textual_gradient: str) -> str:
        """Generate new candidate document"""
        prompt = wrap_textual_update_prompt(best_document, textual_gradient, self.domain)
        response = self.optimizer_model.query(prompt, temperature=self.args.temperature)
        match = re.search(r'Rewritten Document:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            response = response.strip()
        return response
    
    def generate_candidates_parallel(self, best_document: str, textual_gradient: str, 
                                   num_candidates: int, max_workers: int = 4) -> List[str]:
        """Generate multiple candidates in parallel"""
        def generate_single_candidate():
            return self.generate_new_candidate(best_document, textual_gradient)
        
        candidates = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(generate_single_candidate) for _ in range(num_candidates)]
            
            # Collect results
            for future in tqdm(as_completed(futures), total=num_candidates, desc="Generating candidates"):
                candidate = future.result()
                candidates.append(candidate)
        
        return candidates
    
    def evaluate_candidates_parallel(self, candidates: List[str], related_queries: Dict[str, List[str]], 
                                   selected_queries: List[str], original_doc: str,
                                   correct_answer: Optional[str], target_answer: Optional[str],
                                   attack_type: str, num_facts: int = 1,
                                   max_workers: int = 3) -> List[Tuple[float, Dict]]:
        """Evaluate multiple candidates in parallel"""
        def evaluate_single_candidate(candidate: str):
            return self.evaluate_document(
                candidate, related_queries, selected_queries, original_doc,
                correct_answer, target_answer, attack_type, num_facts
            )
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            futures = {executor.submit(evaluate_single_candidate, candidate): i 
                      for i, candidate in enumerate(candidates)}
            
            # Collect results (maintain order)
            candidate_results = [None] * len(candidates)
            for future in tqdm(as_completed(futures), total=len(candidates), desc="Evaluating candidates"):
                candidate_idx = futures[future]
                score, details = future.result()
                candidate_results[candidate_idx] = (score, details)
        
        # Filter out None results
        results = [result for result in candidate_results if result is not None]
        return results
    
    def optimize_document(self, corpus_id: str, corpus_data: Dict[str, Any]) -> str:
        """Optimize a single document"""
        print(f"\n[INFO] Start optimizing document: {corpus_id}")
        
        # Extract data
        initial_document = corpus_data['enhanced_erroneous_corpus']
        original_document = corpus_data['corpus']
        related_queries = corpus_data['related_queries']
        selected_queries = corpus_data.get('selected_queries', [])
        facts_list = corpus_data.get('facts', [])  # Correct facts list
        correct_answer = corpus_data.get('correct_answer', '')
        target_answer = corpus_data.get('target_answer', None)
        
        # Calculate number of facts and attack type
        num_facts = len(facts_list) if facts_list else 1
        attack_type = self.args.attack_type
        
        print(f"[INFO] Attack type: {attack_type}")
        print(f"[INFO] Number of anchor queries: {len(selected_queries)}")
        print(f"[INFO] Number of facts: {len(facts_list)}")
        print(f"[INFO] Correct answer: {correct_answer[:100]}..." if len(correct_answer) > 100 else f"[INFO] Correct answer: {correct_answer}")
        if target_answer:
            print(f"[INFO] Target answer: {target_answer[:100]}..." if len(target_answer) > 100 else f"[INFO] Target answer: {target_answer}")
        else:
            print(f"[INFO] Untargeted attack mode")
        
        # Initialize optimization history
        history = OptimizationHistory(self.domain, self.args.history_size)
        
        # Evaluate initial document
        print("[INFO] Evaluating initial document...")
        initial_score, initial_details = self.evaluate_document(
            initial_document, related_queries, selected_queries, original_document,
            correct_answer, target_answer, attack_type, num_facts
        )
        
        history.add_candidate(
            document=initial_document, total_score=initial_score, 
            similarity_scores=initial_details['similarity_scores'],
            trust_scores=initial_details['trust_scores'],
            generation=0,
        )

        print(f"[INFO] Initial document score: {initial_score:.4f}")


        # Construct simplified version as worst case
        rejected_doc = initial_document[:len(initial_document)//5] + "..."

        initial_reject_score, initial_reject_details = self.evaluate_document(
            rejected_doc, related_queries, selected_queries, original_document,
            correct_answer, target_answer, attack_type, num_facts
        )

        history.add_candidate(
            document=rejected_doc, total_score=initial_reject_score, 
            similarity_scores=initial_reject_details['similarity_scores'],
            trust_scores=initial_reject_details['trust_scores'],
            generation=0,
        )

        print(f"[INFO] Simplified document score: {initial_reject_score:.4f}")

        # Early stopping variables
        no_improve_counter = 0
        last_best_score = initial_score

        # Start iterative optimization
        for iteration in range(self.args.max_iterations):
            print(f"\n[INFO] Iteration {iteration + 1}/{self.args.max_iterations}")
            
            # Get best and worst document pair
            best_doc, worst_doc = history.get_best_worst_pair()
            
            # Generate textual loss
            print("[INFO] Generating textual loss analysis...")
            textual_loss = self.generate_textual_loss(best_doc, worst_doc)

            # Generate textual gradient
            print("[INFO] Generating improvement suggestions...")
            textual_gradient = self.generate_textual_gradient(textual_loss, best_doc)
            
            # Generate new candidates (parallel)
            print(f"[INFO] Generating {self.args.candidates_per_iteration} new candidates in parallel...")
            new_candidates = self.generate_candidates_parallel(
                best_doc['document'], 
                textual_gradient, 
                self.args.candidates_per_iteration,
                max_workers=self.args.generation_workers
            )

            # Evaluate new candidates (parallel)
            print("[INFO] Evaluating new candidates in parallel...")
            evaluation_results = self.evaluate_candidates_parallel(
                new_candidates, related_queries, selected_queries, original_document,
                correct_answer, target_answer, attack_type, num_facts,
                max_workers=self.args.evaluation_workers
            )
            
            # Add to history
            for i, (score, details) in enumerate(evaluation_results):
                history.add_candidate(
                    document=new_candidates[i], total_score=score,
                    similarity_scores=details.get('similarity_scores', {}),
                    trust_scores=details.get('trust_scores', {}),
                    generation=iteration + 1,
                )
                print(f"[INFO] Candidate {i+1} score: {score:.4f}")
                    
            # Show current best result
            current_best = history.get_best_document()
            current_score = current_best['total_score']
            print(f"[INFO] Current best score: {current_score:.4f} (Generation {current_best['generation']})")
            
            # ========== Early Stopping Logic ==========
            if current_score > last_best_score:
                print(f"[INFO] Score improved from {last_best_score:.4f} to {current_score:.4f}. Resetting counter.")
                last_best_score = current_score
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                print(f"[INFO] Score did not improve. Counter: {no_improve_counter}/{self.early_stop_patience}")
            
            if no_improve_counter >= self.early_stop_patience:
                print(f"[INFO] Early stopping triggered after {self.early_stop_patience} iterations without improvement.")
                break

        # Get final result
        final_best = history.get_best_document()
        print(f"[INFO] Optimization completed! Final score: {final_best['total_score']:.4f}")
        
        # Construct full optimization result with metadata
        final_evaluation = {
            'similarity_scores': final_best.get('similarity_scores', {}),
            'trust_scores': final_best.get('trust_scores', {}),
        }
        
        optimization_result = {
            'best_document': final_best['document'],
            'best_score': final_best['total_score'],
            'total_iterations': iteration + 1,
            'early_stopped': no_improve_counter >= self.early_stop_patience,
            'early_stop_counter': no_improve_counter,
            'final_evaluation': final_evaluation,
            'convergence_info': {
                'final_early_stop_counter': no_improve_counter,
                'patience': self.early_stop_patience
            }
        }

        return optimization_result
# ========== Main Function ==========
def main():
    """Main function"""
    args = parse_args()
    print(args)
    print("Only two rewards: Generalization Reward + Misleading Reward")
    
    # Only show status report
    if args.report_only:
        print_optimization_report(args.file_path)
        return
    
    # Load data
    data = load_corpus(args.file_path)
    all_corpus_ids = list(data.keys())
    status = check_optimization_status(data)
    
    # Slice the original ID list, ensuring slice index is not out of bounds
    start_index = max(0, args.start_id - 1)
    end_index = min(len(all_corpus_ids), args.end_id)
    # Get all IDs within the current task range
    target_ids_in_slice = all_corpus_ids[start_index:end_index]
    # Filter items that really need optimization
    items_to_optimize = []
    for corpus_id in target_ids_in_slice:
        if status[corpus_id]['ready_for_optimization'] and not status[corpus_id]['has_optimization']:
            items_to_optimize.append((corpus_id, data[corpus_id]))

    if not items_to_optimize:
        print(f"[INFO] No corpus needs optimization within the specified ID range ({args.start_id}-{args.end_id}).")
        return
    
    total_unoptimized_count = sum(1 for s in status.values() if s['ready_for_optimization'] and not s['has_optimization'])
    print(f"[INFO] Total unoptimized corpus in dataset: {total_unoptimized_count}.")
    print(f"[INFO] Current task range: ID {args.start_id} to {args.end_id}.")
    print(f"[INFO] Will process {len(items_to_optimize)} corpus in this range.")

    # Initialize optimizer
    optimizer = TPOOptimizer(args)
    
    # Optimize corpus one by one
    successful_optimizations = 0
    pending_saves = {}

    for i, (original_corpus_id, original_corpus_data) in enumerate(tqdm(items_to_optimize, desc="Optimization Progress")):
        corpus_id = original_corpus_id
        print(f"\n[INFO] Processing document: {corpus_id} ({i+1}/{len(items_to_optimize)})")

        optimization_result = optimizer.optimize_document(corpus_id, original_corpus_data)
        updated_corpus_data = data[corpus_id]
        updated_corpus_data['final_optimized_corpus'] = optimization_result['best_document']

        updated_corpus_data['optimization_metadata'] = {
            'best_score': optimization_result['best_score'],
            'total_iterations': optimization_result['total_iterations'],
            'early_stopped': optimization_result.get('early_stopped', False),
            'convergence_info': optimization_result.get('convergence_info', {}),
            'final_similarity_scores': {
                'generalization_score': optimization_result['final_evaluation']['similarity_scores'].get('generalization_score', 0.0),
            },
            'final_trust_scores': optimization_result['final_evaluation'].get('trust_scores', {}),
        }

        pending_saves[corpus_id] = updated_corpus_data
        successful_optimizations += 1

        print(f"[SUCCESS] Document {corpus_id} optimized successfully, final score: {optimization_result['best_score']:.4f}")
        print(f"[INFO] {len(pending_saves)} results in pending save buffer.")
                
        is_last_iteration = (i == len(items_to_optimize) - 1)
        should_save_now = (args.save_progress and len(pending_saves) >= 1) or is_last_iteration
        
        if should_save_now:
            print(f"\n[SAVE] Save condition met, saving {len(pending_saves)} documents...")
            save_progress(args.file_path, pending_saves)
            pending_saves.clear()
            print("[SAVE] Buffer cleared.\n")
    
    # Show final status
    print(f"\n=== Optimization Completed ===")
    print(f"Successfully optimized: {successful_optimizations}")
    print(f"Total: {len(items_to_optimize)}")
    print_optimization_report(args.file_path)

if __name__ == "__main__":
    main()
