import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Callable
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from src.gpt_model import GPT
from tqdm import tqdm
import re
from src.pipeline_prompts import (
    fact_extraction_prompt,
    query_generation_prompt,
    error_facts_prompt,
    error_doc_prompt,
    incorporate_queries_prompt,
    targeted_query_generation_prompt,
    targeted_error_facts_prompt,
    targeted_error_doc_prompt,
)

# ========== Argument Parsing ==========
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Adversarial Example Generation Pipeline")
    # File path parameters
    parser.add_argument('--root_path', type=str, default="datasets/", help="Dataset root directory")
    parser.add_argument('--dataset', type=str, default="bioasq", choices=["bioasq", "finqa", "tiebe"])
    parser.add_argument('--version', type=str)
    # Domain parameters
    parser.add_argument('--domain', type=str, default="biomedical", choices=["biomedical", "financial", "real event"])
    # Model parameters
    parser.add_argument('--extraction_model', type=str, default="gpt-4o-mini")
    parser.add_argument('--query_model', type=str, default="gemini-2.5-flash")
    parser.add_argument('--error_model', type=str, default="gemini-2.5-flash")
    parser.add_argument('--incorporation_model', type=str, default="o4-mini")
    # Attack type parameters
    parser.add_argument('--attack_type', type=str, default="targeted", choices=["targeted", "untargeted"], 
                        help="Attack type: targeted or untargeted")
    # Execution parameters
    parser.add_argument('--per_num_queries', type=int, default=3, help="Number of queries generated per role")
    parser.add_argument('--max_workers', type=int, default=5, help="Maximum worker threads for parallel processing")
    parser.add_argument('--steps', type=lambda s: s.split(','), default="all",
                        help="Steps to execute, options: facts, queries, erroneous, incorporation, all")
    parser.add_argument('--save_progress', action='store_true', default=True, help="Whether to save intermediate progress")
    # Display parameters
    parser.add_argument('--report_only', action='store_true', help="Only show status report, do not execute generation")
    
    args = parser.parse_args()
    # Process file path
    suffix = "targeted" if args.attack_type == "targeted" else "untargeted"
    args.file_path = os.path.join(args.root_path, f"{args.version}/{args.dataset}-{args.version}-{suffix}.json")

    return args

# ========== Core Utility Functions ==========
def load_corpus(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load corpus data from file and return dictionary."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Creating new file.")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def check_existing_data(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    """Check existing data for each corpus ID."""
    status = {}
    for corpus_id, corpus_data in data.items():
        status[corpus_id] = {
            'has_facts': 'facts' in corpus_data and corpus_data['facts'],
            'has_queries': 'related_queries' in corpus_data and corpus_data['related_queries'],
            'has_erroneous_facts': 'erroneous_facts' in corpus_data and corpus_data['erroneous_facts'],
            'has_erroneous_corpus': 'erroneous_corpus' in corpus_data and corpus_data['erroneous_corpus'],
            'has_enhanced_corpus': 'enhanced_erroneous_corpus' in corpus_data and corpus_data['enhanced_erroneous_corpus'],
            'has_selected_queries': 'selected_queries' in corpus_data and corpus_data['selected_queries']
        }
    return status

def save_progress(file_path: str, data: Dict[str, Dict[str, Any]]):
    """Save current progress to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Progress saved to {file_path}")

# ========== General Processing Functions ==========
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_with_retry(func: Callable, model: GPT, *args, **kwargs):
    try:
        result = func(model, *args, **kwargs)
        time.sleep(1)
        return result
    except Exception as e:
        print(f"Error processing with {model.model_name}: {str(e)}")
        raise

def parallel_process(process_func: Callable, items_to_process: List[Tuple], 
                    max_workers: int, desc: str) -> Dict[str, Any]:
    """General parallel processing function."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_func, *item): item[0]  # Assume first element is ID
            for item in items_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                if result:
                    results[result[0]] = result[1]  # Assume result is (id, result_data)
            except Exception as e:
                item_id = futures[future]
                print(f"Failed processing {item_id}: {str(e)}")

    return results

# ========== Core Processing Functions ==========
def extract_facts_with_retry(model: GPT, corpus: str, domain: str) -> List[str]:
    """Extract FACTs from corpus"""
    prompt = fact_extraction_prompt(corpus, domain)
    response = model.query(prompt, temperature=0.7)

    match = re.search(r'Extracted assertions:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        response = response.strip()

    # Extract FACTs from response
    facts = response.split('\n')
    # Clean up (remove numbers, extra spaces, etc.)
    facts = [f.strip().lstrip('1234567890.- ') for f in facts if f.strip()]

    # Validate result - empty result will trigger retry
    if not facts or len(facts) == 0:
        raise ValueError(f"Extracted facts are empty, original response: {response[:200]}...")
    
    # Check facts quality
    valid_facts = [f for f in facts if f and len(f.strip()) > 10]
    if len(valid_facts) == 0:
        raise ValueError(f"Extracted facts quality is low, all facts are too short: {facts}")
    
    return valid_facts

def generate_queries_with_retry(model: GPT, corpus: str, role: str, domain: str, 
                               attack_type: str, num_queries: int = 3, 
                               key_assertion: str = None, correct_answer: str = None) -> List[str]:
    """Generate diverse queries from specific role perspective, supports targeted and untargeted attacks"""
    if attack_type == "targeted":
        if correct_answer is None:
            raise ValueError("Targeted attack requires correct_answer parameter")
        prompt = targeted_query_generation_prompt(corpus, role, domain, correct_answer, num_queries)
    else:  # untargeted
        if key_assertion is None:
            raise ValueError("Untargeted attack requires key_assertion parameter")
        prompt = query_generation_prompt(corpus, role, domain, key_assertion, num_queries)
    
    response = model.query(prompt, temperature=1.0)
    
    match = re.search(r'Questions:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        response = response.strip()

    # Extract questions from response
    questions = response.split('\n')
    # Clean up questions (remove numbers, extra spaces, etc.)
    questions = [q.strip().lstrip('1234567890.- ') for q in questions if q.strip()]
    
    # Validate result - empty result will trigger retry
    if not questions or len(questions) == 0:
        raise ValueError(f"Generated queries are empty, role: {role}, original response: {response[:200]}...")
    
    # Check queries quality
    valid_questions = [q for q in questions if q and len(q.strip()) > 10]
    if len(valid_questions) == 0:
        raise ValueError(f"Generated queries quality is low, all queries are too short, role: {role}, queries: {questions}")
    
    return valid_questions[:num_queries]

def generate_erroneous_facts(model: GPT, corpus: str, facts: List[str], domain: str, 
                            attack_type: str = "untargeted", target_answer: str = None) -> List[str]:
    """Generate erroneous fact list, with iterative check and refinement"""
    facts_str = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))
    
    # Generate initial erroneous facts
    if attack_type == "targeted":
        if target_answer is None:
            raise ValueError("Targeted attack requires target_answer parameter")
        initial_prompt = targeted_error_facts_prompt(corpus, facts_str, target_answer, domain)
    else:  # untargeted
        initial_prompt = error_facts_prompt(corpus, facts_str, domain)
    
    response = model.query(initial_prompt, temperature=0.7)
    match = re.search(r'Final Assertion Set:\s*(.*)', response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        response = response.strip()
    current_erroneous_facts = response.split('\n')
    current_erroneous_facts = [f.strip().lstrip('1234567890.- ') for f in current_erroneous_facts if f.strip()]

    # Validate result - empty result will trigger retry
    if not current_erroneous_facts or len(current_erroneous_facts) == 0:
        raise ValueError(f"Generated erroneous facts are empty, attack type: {attack_type}, original response: {response[:200]}...")
    
    # Check erroneous facts quality
    valid_erroneous_facts = [f for f in current_erroneous_facts if f and len(f.strip()) > 15]
    if len(valid_erroneous_facts) == 0:
        raise ValueError(f"Generated erroneous facts quality is low, all facts are too short, attack type: {attack_type}, facts: {current_erroneous_facts}")

    return valid_erroneous_facts

def generate_erroneous_document(model: GPT, original_corpus: str, erroneous_facts: List[str], 
                               domain: str, attack_type: str = "untargeted", target_answer: str = None) -> str:
    """Generate erroneous document based on erroneous facts, supports targeted and untargeted attacks, with iterative check"""
    facts_sheet_str = "\n".join([f"{idx + 1}. {fact}" for idx, fact in enumerate(erroneous_facts)])
    
    # Select initial generation prompt
    if attack_type == "targeted":
        if target_answer is None:
            raise ValueError("Targeted attack requires target_answer parameter")
        initial_prompt = targeted_error_doc_prompt(original_corpus, facts_sheet_str, target_answer, domain)
    else:  # untargeted
        initial_prompt = error_doc_prompt(original_corpus, facts_sheet_str, domain)

    # Generate initial document
    current_corpus = model.query(initial_prompt, temperature=0.7)
    match = re.search(r'Revised document:\s*(.*)', current_corpus, flags=re.IGNORECASE | re.DOTALL)
    if match:
        current_corpus = match.group(1).strip()
    else:
        current_corpus = current_corpus.strip()

    # Validate result - empty result will trigger retry
    if not current_corpus or len(current_corpus) == 0:
        raise ValueError(f"Generated erroneous document is empty, attack type: {attack_type}")
    
    # Check document quality
    if len(current_corpus) < 50:
        raise ValueError(f"Generated erroneous document is too short ({len(current_corpus)} chars), attack type: {attack_type}, document: {current_corpus[:100]}...")

    return current_corpus

def incorporate_queries(model: GPT, erroneous_corpus: str, queries_by_role: Dict[str, List[str]], 
                        facts: List[str], domain: str, per_num_queries: int, attack_type: str = "untargeted") -> Tuple[str, List[str]]:
    """Naturally incorporate queries into erroneous document, supports different selection strategies for targeted and untargeted attacks"""
    if not queries_by_role:
        raise ValueError("No queries to incorporate")
    
    selected_queries = []
    roles = ["novice", "learner", "explorer", "critic", "expert", "analyst"]
    
    if attack_type == "targeted":
        # Targeted attack: Randomly select one query from each role
        print("Using targeted attack query selection strategy: Randomly select one query from each role")
        for role in roles:
            role_queries = queries_by_role.get(role, [])
            if role_queries:
                selected_query = random.choice(role_queries)
                selected_queries.append(selected_query)
    
    else:  # untargeted
        # Untargeted attack: Rotate selection based on facts
        print("Using untargeted attack query selection strategy: Rotate selection based on facts")
        if not facts:
            raise ValueError("Untargeted attack requires facts parameter for query selection")
        
        # 1. Reorganize query structure: fact -> role -> queries
        # Queries are stored in fact order: in each role's query list, first per_num_queries correspond to first fact
        queries_by_fact_and_role = {}
        num_facts = len(facts)
        
        for fact_idx in range(num_facts):
            queries_by_fact_and_role[fact_idx] = {}
            for role in roles:
                role_queries = queries_by_role.get(role, [])
                start_idx = fact_idx * per_num_queries
                end_idx = start_idx + per_num_queries
                
                if end_idx <= len(role_queries):
                    queries_by_fact_and_role[fact_idx][role] = role_queries[start_idx:end_idx]
        
        # 2. Select queries from different roles for each fact
        # 3. Shuffle order
        random.shuffle(roles)
        
        for fact_idx in range(num_facts):
            role = roles[fact_idx % len(roles)]
            available_queries = queries_by_fact_and_role.get(fact_idx, {}).get(role, [])
            
            if available_queries:
                selected_query = random.choice(available_queries)
                selected_queries.append(selected_query)
            
    
    # Validate selected queries
    if not selected_queries or len(selected_queries) == 0:
        raise ValueError(f"Failed to select any queries for incorporation, attack type: {attack_type}")
    
    # Check queries quality
    valid_selected_queries = [q for q in selected_queries if q and len(q.strip()) > 10]
    if len(valid_selected_queries) == 0:
        raise ValueError(f"Selected queries quality is low, all queries are too short, attack type: {attack_type}, queries: {selected_queries}")
    
    # 4. Incorporate selected queries into erroneous document
    queries_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(valid_selected_queries)])
    print(f"Final selected queries count: {len(valid_selected_queries)} (Attack type: {attack_type})")
    
    prompt = incorporate_queries_prompt(erroneous_corpus, queries_str, domain)
    enhanced_corpus = model.query(prompt, temperature=0.7)

    match = re.search(r'Modified text:\s*(.*)', enhanced_corpus, flags=re.IGNORECASE | re.DOTALL)
    if match:
        enhanced_corpus = match.group(1).strip()
    else:
        enhanced_corpus = enhanced_corpus.strip()
    
    # Validate enhanced document
    if not enhanced_corpus or len(enhanced_corpus) == 0:
        raise ValueError(f"Enhanced document after query incorporation is empty, attack type: {attack_type}")
    
    # Check enhanced document quality
    if len(enhanced_corpus) < 50:
        raise ValueError(f"Enhanced document after query incorporation is too short ({len(enhanced_corpus)} chars), attack type: {attack_type}, document: {enhanced_corpus[:100]}...")
    
    return enhanced_corpus, valid_selected_queries



# ========== Parallel Processing Functions ==========
def process_facts_for_corpus(extraction_model: GPT, 
                           corpus_id: str, corpus_text: str, domain: str) -> Tuple[str, List[str]]:
    """Process facts for a single corpus."""
    all_facts = []
    
    # Extract facts using single model
    try:
        facts = process_with_retry(extract_facts_with_retry, extraction_model, corpus_text, domain)
        all_facts.extend(facts)
    except Exception as e:
        print(f"Failed to extract facts for corpus {corpus_id} using {extraction_model.model_name}: {str(e)}")
    
    # Directly return extracted facts without aggregation
    return corpus_id, all_facts

def process_queries_for_corpus(query_model: GPT, corpus_id: str, corpus_text: str, 
                            domain: str, per_num_queries: int, attack_type: str,
                            facts: List[str] = None, correct_answer: str = None) -> Tuple[str, Dict[str, List[str]]]:
    """Process query generation for a single corpus, supports targeted and untargeted attacks"""
    roles = ["novice", "learner", "explorer", "critic", "expert", "analyst"]
    role_queries = {role: [] for role in roles}
    
    if attack_type == "targeted":
        # Targeted attack: Generate queries for each role based on correct answer
        if correct_answer is None:
            raise ValueError("Targeted attack requires correct_answer parameter")
        for role in roles:
            try:
                queries = process_with_retry(generate_queries_with_retry, query_model, 
                                           corpus_text, role, domain, attack_type, per_num_queries,
                                           correct_answer=correct_answer)
                role_queries[role].extend(queries)
            except Exception as e:
                print(f"Failed to generate targeted {role} queries for corpus {corpus_id}: {str(e)}")
                continue
    else:  # untargeted
        # Untargeted attack: Generate queries for each fact and each role
        if facts is None:
            raise ValueError("Untargeted attack requires facts parameter")
        for fact in facts:
            for role in roles:
                try:
                    queries = process_with_retry(generate_queries_with_retry, query_model, 
                                                corpus_text, role, domain, attack_type, per_num_queries,
                                                key_assertion=fact)
                    role_queries[role].extend(queries)
                except Exception as e:
                    print(f"Failed to generate {role} queries for fact '{fact[:50]}...' in corpus {corpus_id}: {str(e)}")
                    continue
    
    return corpus_id, role_queries

def process_erroneous_doc_for_corpus(error_model: GPT, corpus_id: str, 
                                  corpus_data: Dict[str, Any], domain: str, 
                                  attack_type: str = "untargeted", target_answer: str = None) -> Tuple[str, Dict[str, Any]]:
    """Process erroneous document generation for a single corpus, supports targeted and untargeted attacks"""
    # Step 1: Generate erroneous facts
    erroneous_facts = process_with_retry(generate_erroneous_facts, 
                                        error_model, 
                                        corpus_data['corpus'], 
                                        corpus_data['facts'],
                                        domain,
                                        attack_type,
                                        target_answer,)
    
    # Step 2: Generate erroneous document based on erroneous facts
    erroneous_corpus = process_with_retry(generate_erroneous_document,
                                        error_model,
                                        corpus_data['corpus'],
                                        erroneous_facts,
                                        domain,
                                        attack_type,
                                        target_answer,)
    # Return erroneous facts and erroneous document
    return corpus_id, {
        'erroneous_facts': erroneous_facts,
        'erroneous_corpus': erroneous_corpus
    }

def process_query_incorporation_for_corpus(incorporation_model: GPT, corpus_id: str, 
                                        corpus_data: Dict[str, Any], domain: str, per_num_queries: int, 
                                        attack_type: str = "untargeted") -> Tuple[str, Dict[str, Any]]:
    """Process query incorporation for a single corpus."""
    try:
        enhanced_erroneous_corpus, selected_queries = process_with_retry(incorporate_queries,
                                                      incorporation_model,
                                                      corpus_data['erroneous_corpus'],
                                                      corpus_data['related_queries'],
                                                      corpus_data['facts'],
                                                      domain,
                                                      per_num_queries,
                                                      attack_type)
        return corpus_id, {
            'enhanced_erroneous_corpus': enhanced_erroneous_corpus,
            'selected_queries': selected_queries
        }
    except Exception as e:
        print(f"Failed to incorporate queries for corpus {corpus_id}: {str(e)}")
        return corpus_id, {
            'enhanced_erroneous_corpus': corpus_data['erroneous_corpus'],  # Fallback
            'selected_queries': []
        }

# ========== Pipeline Integration Functions ==========
def run_pipeline_step(step_name: str, data: Dict[str, Dict[str, Any]], 
                    process_func: Callable, filter_func: Callable, 
                    process_args: Dict[str, Any], max_workers: int, 
                    desc: str) -> Dict[str, Dict[str, Any]]:
    """Generic pipeline step execution function."""
    print(f"\n=== Step: {step_name} ===")
    
    # Filter items to process
    items_to_process = filter_func(data)
    
    if not items_to_process:
        print(f"All corpora have completed {step_name} or missing required data. Skipping.")
        return data
    
    print(f"Executing {step_name} for {len(items_to_process)} corpora...")
    
    # Parallel processing
    results = parallel_process(process_func, items_to_process, max_workers, desc)
    
    # Update data
    for corpus_id, result in results.items():
        if result:
            data[corpus_id].update({step_name: result})
    
    return data

def print_status_report(file_path: str):
    """Print detailed status report of current data."""
    data = load_corpus(file_path)
    status = check_existing_data(data)
    
    print("\n=== Status Report ===")
    print(f"Total Corpora: {len(data)}")
    
    # Count completion status
    facts_count = sum(1 for s in status.values() if s['has_facts'])
    queries_count = sum(1 for s in status.values() if s['has_queries'])
    erroneous_facts_count = sum(1 for s in status.values() if s['has_erroneous_facts'])
    erroneous_count = sum(1 for s in status.values() if s['has_erroneous_corpus'])
    enhanced_count = sum(1 for s in status.values() if s['has_enhanced_corpus'])
    selected_queries_count = sum(1 for s in status.values() if s['has_selected_queries'])
    fully_complete = sum(1 for s in status.values() if all(s.values()))
    
    print(f"Fact Extraction: {facts_count}/{len(data)} ({facts_count/len(data)*100:.1f}%)")
    print(f"Query Generation: {queries_count}/{len(data)} ({queries_count/len(data)*100:.1f}%)")
    print(f"Erroneous Facts: {erroneous_facts_count}/{len(data)} ({erroneous_facts_count/len(data)*100:.1f}%)")
    print(f"Erroneous Document: {erroneous_count}/{len(data)} ({erroneous_count/len(data)*100:.1f}%)")
    print(f"Enhanced Document: {enhanced_count}/{len(data)} ({enhanced_count/len(data)*100:.1f}%)")
    print(f"Selected Queries: {selected_queries_count}/{len(data)} ({selected_queries_count/len(data)*100:.1f}%)")
    print(f"Fully Complete: {fully_complete}/{len(data)} ({fully_complete/len(data)*100:.1f}%)")
    
    # Show incomplete items
    incomplete_items = [corpus_id for corpus_id, s in status.items() if not all(s.values())]
    if incomplete_items:
        print(f"\nIncomplete Items: {len(incomplete_items)}")
        for corpus_id in incomplete_items[:5]:  # Show first 5
            s = status[corpus_id]
            missing = [step for step, done in s.items() if not done]
            print(f"  {corpus_id}: Missing {', '.join(missing)}")
        if len(incomplete_items) > 5:
            print(f"  ... and {len(incomplete_items) - 5} more items")
    print("==================\n")

# ========== Main Function ==========
def main():
    """Main function, executes flexible pipeline using command line arguments."""
    args = parse_args()
    
    # Only show status report
    if args.report_only:
        print_status_report(args.file_path)
        return
    
    # Initialize GPT models
    extraction_model = GPT(args.extraction_model, base_url="", api_key="")
    query_model = GPT(args.query_model, base_url="", api_key="")
    error_generation_model = GPT(args.error_model, base_url="", api_key="")
    query_incorporation_model = GPT(args.incorporation_model, base_url="", api_key="")
    
    # Show initial status
    print("=== Initial Status ===")
    print_status_report(args.file_path)
    
    # Load data
    data = load_corpus(args.file_path)
    
    # Determine steps to run
    steps_to_run = args.steps
    if steps_to_run == ['all']:
        steps_to_run = ['facts', 'queries', 'erroneous', 'incorporation']
    
    # Execute selected steps
    if 'facts' in steps_to_run:
        # Fact extraction step
        status = check_existing_data(data)
        items_to_process = [(corpus_id, corpus_data['corpus']) 
                          for corpus_id, corpus_data in data.items() 
                          if 'corpus' in corpus_data and not status[corpus_id]['has_facts']]
        
        # Process fact extraction
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for corpus_id, corpus_text in items_to_process:
                future = executor.submit(process_facts_for_corpus, 
                                       extraction_model, 
                                       corpus_id, corpus_text, args.domain)
                futures[future] = corpus_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Facts"):
                try:
                    corpus_id, facts = future.result()
                    data[corpus_id]['facts'] = facts
                except Exception as e:
                    corpus_id = futures[future]
                    print(f"Failed to process facts for corpus {corpus_id}: {str(e)}")
        
        if args.save_progress:
            save_progress(args.file_path, data)
    
    if 'queries' in steps_to_run:
        # Query generation step
        status = check_existing_data(data)
        items_to_process = [(corpus_id, corpus_data['corpus'], corpus_data.get('facts')) 
                          for corpus_id, corpus_data in data.items() 
                          if 'corpus' in corpus_data 
                          and not status[corpus_id]['has_queries']]
        
        # Process query generation
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for corpus_id, corpus_text, facts in items_to_process:
                corpus_data = data[corpus_id]
                
                if args.attack_type == "targeted":
                    # Targeted attack: Need correct answer from data
                    if 'correct_answer' not in corpus_data:
                        print(f"Warning: Corpus {corpus_id} missing 'correct answer' field, skipping targeted query generation")
                        continue
                    correct_answer = corpus_data['correct_answer']
                    future = executor.submit(process_queries_for_corpus, 
                                           query_model, corpus_id, corpus_text, 
                                           args.domain, args.per_num_queries, args.attack_type,
                                           correct_answer=correct_answer)
                else:
                    # Untargeted attack: Need facts
                    if not facts:
                        print(f"Warning: Corpus {corpus_id} missing facts, skipping untargeted query generation")
                        continue
                    future = executor.submit(process_queries_for_corpus, 
                                           query_model, corpus_id, corpus_text, 
                                           args.domain, args.per_num_queries, args.attack_type,
                                           facts=facts)
                futures[future] = corpus_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Queries"):
                try:
                    corpus_id, queries = future.result()
                    data[corpus_id]['related_queries'] = queries
                except Exception as e:
                    corpus_id = futures[future]
                    print(f"Failed to process queries for corpus {corpus_id}: {str(e)}")
        
        if args.save_progress:
            save_progress(args.file_path, data)

    if 'erroneous' in steps_to_run:
        # Erroneous document generation step
        status = check_existing_data(data)
        items_to_process = [(corpus_id, corpus_data) 
                          for corpus_id, corpus_data in data.items() 
                          if (not status[corpus_id]['has_erroneous_corpus'] 
                              or not status[corpus_id]['has_erroneous_facts'])
                          and status[corpus_id]['has_facts'] 
                          and 'corpus' in corpus_data]
        
        # Process erroneous document generation
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for corpus_id, corpus_data in items_to_process:
                if args.attack_type == "targeted":
                    # Targeted attack: Need target answer
                    if 'target_answer' not in corpus_data:
                        print(f"Warning: Corpus {corpus_id} missing target_answer field, skipping targeted erroneous document generation")
                        continue
                    target_answer = corpus_data['target_answer']
                    future = executor.submit(process_erroneous_doc_for_corpus, 
                                           error_generation_model, corpus_id, 
                                           corpus_data, args.domain, args.attack_type, target_answer)
                else:
                    # Untargeted attack
                    future = executor.submit(process_erroneous_doc_for_corpus, 
                                           error_generation_model, corpus_id, 
                                           corpus_data, args.domain, args.attack_type, None)
                futures[future] = corpus_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Erroneous Documents"):
                try:
                    corpus_id, erroneous_data = future.result()
                    data[corpus_id]['erroneous_facts'] = erroneous_data['erroneous_facts']
                    data[corpus_id]['erroneous_corpus'] = erroneous_data['erroneous_corpus']
                except Exception as e:
                    corpus_id = futures[future]
                    print(f"Failed to process erroneous document for corpus {corpus_id}: {str(e)}")
        
        if args.save_progress:
            save_progress(args.file_path, data)

    if 'incorporation' in steps_to_run:
        # Query incorporation step
        status = check_existing_data(data)
        if args.attack_type == "targeted":
            # Targeted attack: Facts not needed
            items_to_process = [(corpus_id, corpus_data) 
                              for corpus_id, corpus_data in data.items() 
                              if not status[corpus_id]['has_enhanced_corpus'] 
                              and status[corpus_id]['has_erroneous_corpus'] 
                              and status[corpus_id]['has_queries']]
        else:
            # Untargeted attack: Facts needed
            items_to_process = [(corpus_id, corpus_data) 
                              for corpus_id, corpus_data in data.items() 
                              if not status[corpus_id]['has_enhanced_corpus'] 
                              and status[corpus_id]['has_erroneous_corpus'] 
                              and status[corpus_id]['has_queries']
                              and status[corpus_id]['has_facts']]
        
        # Process query incorporation
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for corpus_id, corpus_data in items_to_process:
                future = executor.submit(process_query_incorporation_for_corpus, 
                                       query_incorporation_model, corpus_id, 
                                       corpus_data, args.domain, args.per_num_queries, args.attack_type)
                futures[future] = corpus_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Incorporating Queries"):
                try:
                    corpus_id, result_data = future.result()
                    data[corpus_id]['enhanced_erroneous_corpus'] = result_data['enhanced_erroneous_corpus']
                    data[corpus_id]['selected_queries'] = result_data['selected_queries']
                except Exception as e:
                    corpus_id = futures[future]
                    print(f"Failed to incorporate queries for corpus {corpus_id}: {str(e)}")
        
        if args.save_progress:
            save_progress(args.file_path, data)
    
    # Show final status
    print("=== Final Status ===")
    print_status_report(args.file_path)

if __name__ == "__main__":
    main()