# MIRAGE

Official codebase and supplementary material for our paper: **MIRAGE: Misleading Retrieval-Augmented Generation via Black-box and Query-agnostic Poisoning Attacks**.

## File Structure

```
.
├── datasets/               # Datasets (BioASQ, FinQA, TiEBe) and 1k sampled experimental data (startpoint)
├── src/                    # Source code directory
│   ├── gpt_model.py        # GPT model wrapper and interaction logic
│   ├── nli_judge.py        # NLI judgment logic
│   ├── optmize_prompt.py   # Prompts used for the Phase 3: Adversarial Alignment
│   └── pipeline_prompt.py  # Prompts used for the Phase 1 & 2 pipeline
├── eval.py                 # Main evaluation script (RSR, ASR_L, ASR_S)
├── eval_nli.py             # Evaluation script for Natural Language Inference metrics (ASR_N)
├── eval_stealthiness.py    # Evaluation script for stealthiness metrics (SR)
├── optmize.py              # Script for Phase 3: Adversarial Alignment
├── pipeline.py             # Main pipeline script for Phase 1: Query Distribution Modeling & Phase 2: Semantic Anchoring
├── run.sh                  # Shell script
└── Full_Paper.pdf          # Full research paper
```

## Usage
```bash
bash run.sh
```