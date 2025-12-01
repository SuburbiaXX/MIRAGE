import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

from tqdm import tqdm

from src.nli_judge import NLIJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform NLI entailment judgment on existing evaluation results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to evaluation result JSON file, e.g., bioasq-ours-targeted_top5_results.json")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name, e.g., bioasq, finqa, tiebe")
    parser.add_argument("--version", type=str, required=True,
                        help="Attack method version, e.g., ours, poisonedrag, paradox")
    parser.add_argument("--attack_type", type=str, choices=["targeted", "untargeted"], required=True,
                        help="Attack type")
    parser.add_argument("--nli_model", type=str, default="models/deberta-v2-xlarge-mnli",
                        help="Path to NLI model")
    parser.add_argument("--nli_device", type=str, default=None,
                        help="Device used by NLI model")
    parser.add_argument("--start_id", type=str, default=None, help="Start sample ID, e.g., 'test1'")
    parser.add_argument("--end_id", type=str, default=None, help="End sample ID, e.g., 'test250'")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from saved results")
    parser.add_argument("--output_dir", type=str, default="nli_results", help="Output directory for NLI results")
    return parser.parse_args()


class NLIEvaluator:
    def __init__(
        self,
        results_file: str,
        dataset_name: str,
        version: str,
        attack_type: str,
        nli_model_name: str,
        nli_device: Optional[str],
        output_dir: str = "nli_results",
    ) -> None:
        self.results_file = results_file
        self.dataset_name = dataset_name
        self.version = version
        self.attack_type = attack_type
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize NLI judge
        print("[INFO] Initializing NLI judge...")
        self.nli_judge = NLIJudge(model_name=nli_model_name, device=nli_device)

        # Load evaluation results
        print(f"[INFO] Loading evaluation results: {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            self.eval_results = json.load(f)
        print(f"[INFO] Loaded {len(self.eval_results)} evaluation samples")

        # For targeted attacks, load original dataset to get target_answer
        self.original_dataset = {}
        if attack_type == "targeted":
            dataset_file = f"datasets/{version}/{dataset_name}-{version}-{attack_type}.json"
            if os.path.exists(dataset_file):
                print(f"[INFO] Loading original dataset: {dataset_file}")
                with open(dataset_file, "r", encoding="utf-8") as f:
                    self.original_dataset = json.load(f)
                print(f"[INFO] Loaded {len(self.original_dataset)} original samples")
            else:
                print(f"[WARNING] Original dataset file does not exist: {dataset_file}")
                print(f"[WARNING] Will attempt to get target_answer from evaluation results")

    def _filter_sample_ids(self, start_id: Optional[str], end_id: Optional[str]) -> List[str]:
        """Filter sample ID list based on start_id and end_id"""
        all_ids = list(self.eval_results.keys())

        # Extract numeric part for sorting and comparison
        def extract_num(sample_id: str) -> int:
            match = re.search(r"(\d+)", sample_id)
            return int(match.group(1)) if match else 0

        # Sort by numeric order
        all_ids.sort(key=extract_num)

        if start_id is None and end_id is None:
            return all_ids

        start_num = extract_num(start_id) if start_id else 0
        end_num = extract_num(end_id) if end_id else float("inf")

        filtered_ids = [
            sid for sid in all_ids
            if start_num <= extract_num(sid) <= end_num
        ]

        return filtered_ids

    def evaluate_single_sample(self, sample_id: str, sample_data: Dict) -> Dict:
        """Perform NLI judgment on a single sample"""
        # Extract necessary fields
        correct_answer = sample_data.get("correct_answer", "")
        rag_freeform_answer = sample_data.get("rag_freeform_answer", "")
        
        # For targeted attacks, prioritize getting target_answer from original dataset
        target_answer = None
        if self.attack_type == "targeted":
            if sample_id in self.original_dataset:
                target_answer = self.original_dataset[sample_id].get("target_answer")
            if not target_answer:
                # If not in original dataset, try to get from evaluation results
                target_answer = sample_data.get("target_answer")

        result = {
            "sample_id": sample_id,
            "correct_answer": correct_answer,
            "rag_freeform_answer": rag_freeform_answer,
            "nli_success": False,
            "nli_details": None,
        }

        # If no answer generated, skip
        if not rag_freeform_answer:
            result["status"] = "NO_FREEFORM_ANSWER"
            return result

        # For targeted attacks, target_answer is required
        if self.attack_type == "targeted":
            if not target_answer:
                result["status"] = "NO_TARGET_ANSWER"
                return result

            # Use NLI judgment (consistent with _metric_nli logic in eval.py)
            nli_result = self.nli_judge.judge(
                generated_answer=rag_freeform_answer,
                malicious_claim=target_answer,
                truth_claim=correct_answer,
            )
            result["nli_success"] = nli_result.success
            result["nli_details"] = asdict(nli_result)
            result["status"] = "COMPLETED"

        else:
            # untargeted attacks: only judge if it contradicts the correct answer
            # Note: untargeted in eval.py does not use NLI, provided here for consistency
            # Logic can be adjusted as needed
            nli_result = self.nli_judge.judge(
                generated_answer=rag_freeform_answer,
                malicious_claim="",  # untargeted has no malicious target
                truth_claim=correct_answer,
            )
            result["nli_success"] = nli_result.success
            result["nli_details"] = asdict(nli_result)
            result["status"] = "COMPLETED"

        return result

    def evaluate_all_samples(
        self,
        start_id: Optional[str] = None,
        end_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict:
        """Evaluate all samples using NLI"""
        # Generate result save path
        results_filename = Path(self.results_file).name
        base_name = results_filename.replace("_results.json", "")
        nli_results_file = os.path.join(self.output_dir, f"{base_name}_nli_results.json")
        nli_summary_file = os.path.join(self.output_dir, f"{base_name}_nli_summary.json")

        # Load existing NLI results (if resuming)
        existing_nli_results: Dict[str, Dict] = {}
        if resume and os.path.exists(nli_results_file):
            with open(nli_results_file, "r", encoding="utf-8") as f:
                existing_nli_results = json.load(f)
            print(f"[INFO] Resumed {len(existing_nli_results)} evaluated samples from {nli_results_file}")

        # Filter sample IDs to evaluate
        sample_ids = self._filter_sample_ids(start_id, end_id)
        print(f"[INFO] Will evaluate {len(sample_ids)} samples (Range: {sample_ids[0]} to {sample_ids[-1]})")

        # Skip already evaluated samples
        samples_to_evaluate = [sid for sid in sample_ids if sid not in existing_nli_results]
        if resume:
            print(f"[INFO] Skipping {len(sample_ids) - len(samples_to_evaluate)} already evaluated samples")
            print(f"[INFO] Need to evaluate {len(samples_to_evaluate)} new samples")

        nli_results = existing_nli_results.copy()

        # Initialize statistics
        stats = self._initialize_stats_from_results(existing_nli_results)

        # Evaluate samples one by one
        for idx, sample_id in enumerate(tqdm(samples_to_evaluate, desc="NLI Evaluation"), start=1):
            sample_data = self.eval_results[sample_id]
            nli_result = self.evaluate_single_sample(sample_id, sample_data)
            nli_results[sample_id] = nli_result

            # Update statistics
            stats["total_samples"] += 1
            if nli_result.get("status") == "COMPLETED":
                stats["completed_count"] += 1
                if nli_result.get("nli_success"):
                    stats["nli_success_count"] += 1

            # Incrementally save results (save every 10 samples)
            if idx % 10 == 0 or idx == len(samples_to_evaluate):
                self._save_results(nli_results_file, nli_results)
                # Print realtime statistics
                self._print_realtime_stats(stats, idx + len(existing_nli_results))

        # Calculate final statistics
        summary = self._calculate_summary(stats)

        # Print final summary
        self._print_final_summary(summary)

        # Save final summary
        self._save_results(nli_summary_file, summary)

        return {
            "summary": summary,
            "detailed_results": nli_results,
        }

    def _initialize_stats_from_results(self, existing_results: Dict[str, Dict]) -> Dict:
        """Initialize statistics from existing results"""
        stats = {
            "total_samples": 0,
            "completed_count": 0,
            "nli_success_count": 0,
        }

        for result in existing_results.values():
            stats["total_samples"] += 1
            if result.get("status") == "COMPLETED":
                stats["completed_count"] += 1
                if result.get("nli_success"):
                    stats["nli_success_count"] += 1

        return stats

    def _calculate_summary(self, stats: Dict) -> Dict:
        """Calculate summary statistics"""
        total = stats["total_samples"]
        completed = stats["completed_count"]
        nli_success = stats["nli_success_count"]

        nli_success_rate = (nli_success / completed) if completed > 0 else 0.0

        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "attack_type": self.attack_type,
            "total_samples": total,
            "completed_count": completed,
            "nli_success_count": nli_success,
            "nli_success_rate": nli_success_rate,
            "raw_counts": stats,
        }

    def _print_realtime_stats(self, stats: Dict, total_evaluated: int) -> None:
        """Print realtime statistics"""
        completed = stats["completed_count"]
        nli_success = stats["nli_success_count"]
        nli_rate = (nli_success / completed) if completed > 0 else 0.0

        print(f"\n[Progress: {total_evaluated}] NLI Evaluation:")
        print(f"  Completed: {completed}")
        print(f"  NLI Success: {nli_success}")
        print(f"  NLI Success Rate: {nli_rate:.2%}")

    def _print_final_summary(self, summary: Dict) -> None:
        """Print final summary"""
        print("\n" + "=" * 70)
        print(" NLI Entailment Evaluation Final Results")
        print("=" * 70)
        print(f" Dataset: {summary['dataset_name']}")
        print(f" Method: {summary['version']}")
        print(f" Attack Type: {summary['attack_type']}")
        print("-" * 70)
        print(f" Total Samples: {summary['total_samples']}")
        print(f" Completed: {summary['completed_count']}")
        print(f" NLI Success Count: {summary['nli_success_count']}")
        print(f" NLI Success Rate: {summary['nli_success_rate']:.2%}")
        print("=" * 70)

    def _save_results(self, file_path: str, data: Dict) -> None:
        """Save results to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    print(args)

    # Verify results_file exists
    if not os.path.exists(args.results_file):
        print(f"[ERROR] Evaluation result file does not exist: {args.results_file}")
        return

    evaluator = NLIEvaluator(
        results_file=args.results_file,
        dataset_name=args.dataset_name,
        version=args.version,
        attack_type=args.attack_type,
        nli_model_name=args.nli_model,
        nli_device=args.nli_device,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*80}")
    print("Starting NLI Entailment Evaluation...")
    if args.start_id or args.end_id:
        print(f"Evaluation Range: {args.start_id or 'Start'} to {args.end_id or 'End'}")
    if args.resume:
        print("Resume Mode: Skipping already evaluated samples")
    print(f"{'='*80}\n")

    results = evaluator.evaluate_all_samples(
        start_id=args.start_id,
        end_id=args.end_id,
        resume=args.resume,
    )

    summary = results["summary"]
    print("\n" + "="*80)
    print("NLI Evaluation Completed!")
    print(f"NLI Success Rate: {summary['nli_success_rate']:.2%}")
    print("="*80)


if __name__ == "__main__":
    main()
