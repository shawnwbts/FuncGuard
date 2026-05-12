# FuncGuard: Function-level Smart Contract Generation through Context-aware and Self-verifying LLM Agents

This is the source code and dataset to the paper "FuncGuard: Function-level Smart Contract Generation through Context-aware and Self-verifying LLM Agents". Please refer to the paper for the experimental details.

# Approach

![绘图1](https://github.com/shawnwbts/FuncGuard/blob/master/fig1.png)

## 🎯 Project Scope

This directory provides a **reproducible and extensible** evaluation pipeline for assessing single-function-level Solidity implementations:

- **Input**: `tasks` (`task_id`, `source_id`, `full_signature`, `prompt`) + `key.json` (mapping from `source_id` to test files)
- **Core workflow**: Create an isolated Foundry workspace for each task → replace the target function under range-locked constraints → run `forge build/test` → run `slither` → generate structured reports
- **Output**: Files such as `forge_test_report.json`, `slither_after.json`, `slither_baseline.json`, `slither_layered_report.json`, and `token_usage_rawmodel.json` under `output/task_<id>/`

 ## 📂 Directory Structure (Key Files):

Download the original dataset file [repository.zip](https://media.githubusercontent.com/media/pzy2000/SolEval/refs/heads/master/repository.zip) and extract it into the `sol_fn_miniloop/root/` folder.

In the data folder, we store the data sets after processing, including the training sets and test sets of code, comments.

```
SIRCOT
├── ms_agent  # source code from MS-AGENT
├── eval_alltests.py                   # Evaluate compile@k / pass@k and other metrics for all tasks
├── eval_metrics.py                    # Auxiliary metric aggregation (used together with eval_alltests)
├── projects/sol_fn_miniloop/
    ├── callbacks/                     # Evaluation callbacks/tools reused by baseline scripts
    │   ├── forge_callback.py          # Run Forge, parse outputs, and generate reports
    │   └── security_callback.py       # Run Slither, extract High/Medium findings, and compute diffs
    │   └── artifact_callback.py       # Handles code artifact integration with range-locked patching and validation
    │   ├── task_prep_callback.py      # Prepares isolated Foundry workspaces and task-specific context for evaluation
    │   ├── test_select_callback.py    # Selects relevant test functions for the target function and prepares test execution context
    ├── root/                          # Original contract repositories copied into workspaces
├── key.json                   # Mapping from source_id to test files 
├── example_task.json                   # Code generation task dataset
```

## 🚀 Quick Start

### 1) Prepare Inputs

- `example_task.json` (or the `tasks` argument): each task should contain at least `task_id`, `source_id`, `full_signature`, and `prompt`
- `key.json`: mapping from `source_id` to test file paths

### 2) Generate Functions 

```
python batch_cli.py
```

Output: multiple generated function files under `output/task_<id>/`.

### 3) Aggregate Evaluation Metrics

```
# Compute compile@k / pass@k (k=1)
python eval_alltests.py --tasks example_task.json --output-dir output 

# Compute security@k  (k=1)
python eval_metrics.py --tasks example_task.json --recompute-slither
```

