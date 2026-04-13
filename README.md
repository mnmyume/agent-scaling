# Agent Scaling

A framework for studying scaling behaviors of LLM-based single-agent and multi-agent systems on complex reasoning tasks.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling

# Install dependencies
uv sync --prerelease=allow

# Install flash-attn (needed for BrowseComp+ environment)
uv pip install --no-build-isolation flash-attn

# Activate the virtual environment
source .venv/bin/activate
```

### Setting Environment Variables

Create a `.env` file with your LLM API keys. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for supported providers.

```bash
# Required: At least one LLM provider API key
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key"
ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: LangFuse for LLM call tracing
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY="your-secret-key"
LANGFUSE_PUBLIC_KEY="your-public-key"
```

## Running Experiments

All commands below work with the original `python run_scripts/run_experiment.py ...` interface after `source .venv/bin/activate`. If you prefer not to activate the virtual environment, prepend `uv run`.

### Basic Usage

```bash
python run_scripts/run_experiment.py
python run_scripts/run_experiment.py debug=true max_instances=1
```

The framework uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, so any field can be overridden from the CLI:

```bash
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test
python run_scripts/run_experiment.py llm.model=gpt-4o-mini
python run_scripts/run_experiment.py num_workers=4
python run_scripts/run_experiment.py token_budget.total_tokens_per_instance=4800
```

### Implemented Agent Configs

| Architecture | Config Name | Description |
|-------------|-------------|-------------|
| Single-Agent | `single-agent` | Single LLM agent with tool use |
| Independent MAS | `multi-agent-independent` | Parallel workers with no live coordination |
| Decentralized MAS | `multi-agent-decentralized` | Peer-to-peer coordination without a lead agent |
| Centralized MAS | `multi-agent-centralized` | Lead-agent orchestration with worker agents |
| Hybrid MAS | `multi-agent-hybrid` | Lead-agent orchestration plus bounded peer exchange |

### Implemented Dataset Selectors

| Dataset | Config Name | Notes |
|---------|-------------|-------|
| PlanCraft test subset | `plancraft-test` | Objective environment-grounded evaluation |
| BrowseComp+ sampled subset | `browsecomp-plus` | LLM-graded via the reusable BrowseComp grader |
| WorkBench full set | `workbench` | Deterministic state-based evaluation via tool-call replay; no LLM grader required |
| WorkBench analytics split | `workbench-analytics` | Same deterministic WorkBench evaluator |
| WorkBench calendar split | `workbench-calendar` | Same deterministic WorkBench evaluator |
| WorkBench CRM split | `workbench-customer-relationship-manager` | Same deterministic WorkBench evaluator |
| WorkBench email split | `workbench-email` | Same deterministic WorkBench evaluator |
| WorkBench multi-domain split | `workbench-multi-domain` | Same deterministic WorkBench evaluator |
| WorkBench project-management split | `workbench-project-management` | Same deterministic WorkBench evaluator |

WorkBench is now wired into `run_conf/dataset/`. Unlike BrowseComp+, it does not use `eval_llm` or an LLM grader; correctness is computed by replaying the predicted tool calls against the sandbox and comparing the resulting state to the gold state.

#### Supported LLMs

| Provider | Models | Prefix |
|----------|--------|--------|
| OpenAI | GPT-5, GPT-5-mini, GPT-5-nano | (none) |
| Google | Gemini-2.5 Pro, Gemini-2.5 Flash, Gemini-2.0 Flash | `gemini/` |
| Anthropic | Claude 4.5 Sonnet, Claude 4.0 Sonnet, Claude 3.7 Sonnet | `anthropic/` |
| OpenRouter | Any model available on OpenRouter | `openrouter/` |
| Minimax | MiniMax-Text-01 (Anthropic-compatible API) | `minimax/` |

**OpenRouter**: Set `OPENROUTER_API_KEY` in `.env`. Use models like `openrouter/anthropic/claude-3.5-sonnet`.

**Minimax**: Set `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic` in `.env`. Use models like `minimax/MiniMax-Text-01`.

## Example Experiments

### SAS

```bash
python run_scripts/run_experiment.py \
    agent=single-agent \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Centralized

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Independent

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-independent \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Decentralized

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-decentralized \
    dataset=browsecomp-plus \
    llm.model=gpt-4o-mini \
    max_instances=5
```

### Hybrid

```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-hybrid \
    dataset=browsecomp-plus \
    llm.model=gpt-4o-mini \
    max_instances=5
```

### Tiny Smoke Runs

```bash
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-independent dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=single-agent dataset=workbench-email debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=browsecomp-plus debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=browsecomp-plus debug=true max_instances=1
```

### Paper Metrics Calculator

Use the paper metrics calculator after you have finished running the relevant
architectures for a dataset/model pair:

```bash
uv run python run_scripts/materialize_paper_metrics.py exp_outputs/plancraft-test
uv run python run_scripts/materialize_paper_metrics.py exp_outputs/workbench
```

The script does not take a `--model` flag. It infers the model from the run
directories you pass in:

- Pass a dataset root such as `exp_outputs/plancraft-test` to process every
  discovered model under that dataset.
- Pass only model-specific directories to restrict calculation to one model.

Example for only `minimax/MiniMax-M2.7`:

```bash
uv run python run_scripts/materialize_paper_metrics.py \
  exp_outputs/plancraft-test/*/minimax/MiniMax-M2.7
```

What it does:

- Scans the provided experiment directory recursively for completed runs.
- Groups runs by `(dataset_id, model, token_budget)`.
- For each architecture in that group, selects the run with the largest completed
  instance count, breaking ties by newer run timestamp.
- Recomputes paired paper metrics on the shared completed instance set across the
  selected architectures.
- Writes the canonical result under `exp_outputs/<dataset>/paper_metrics/...`.

The calculator only writes a summary when a pairable `single-agent` baseline
exists and the selected runs share a completed instance subset.

Output layout:

```text
exp_outputs/<dataset>/paper_metrics/<provider>/<model>/token_budget_<budget>/paper_metrics_summary.json
```

Example:

```text
exp_outputs/plancraft-test/paper_metrics/minimax/MiniMax-M2.7/token_budget_4800/paper_metrics_summary.json
```

If you want the raw aggregation across every discovered run history instead of
the single canonical paper-metrics output, use:

```bash
uv run python run_scripts/aggregate_metrics.py exp_outputs/plancraft-test
uv run python run_scripts/aggregate_metrics.py exp_outputs/browsecomp_plus_sampled_100
```

That script keeps all discovered runs in one summary and only emits paired
metrics when `single-agent` and MAS runs have exactly matching completed
instance indices.

### Scaling Principle Predictor

Use the scaling-principle predictor after `paper_metrics_summary.json` files
have already been materialized:

```bash
uv run python run_scripts/predict_scaling_principle.py
```

By default, the script:

- Recursively searches `exp_outputs/` for `paper_metrics_summary.json`.
- Filters to `model=minimax/MiniMax-M2.7` and `token_budget=4800`.
- Reads the paper-style inputs from each summary's `paired_metrics` block.
- Applies the hardcoded MiniMax scaling inputs from the script:
  `I=55.0`, `I_mean=56.9`, `T={'workbench': 26, 'plancraft-test': 1}`, and
  `n_a={'single-agent': 1, 'multi-agent-*': 3}`.
- Builds the regression features, including the required `log1p`
  transformations, Table 4 interaction terms, and dataset-wide z-score
  standardization across the parsed rows.
- Writes the comparison CSV to `predict_results/prediction_comparison.csv`.

Example with explicit arguments:

```bash
uv run python run_scripts/predict_scaling_principle.py \
  --exp-root exp_outputs \
  --model minimax/MiniMax-M2.7 \
  --token-budget 4800 \
  --output-dir predict_results
```

Useful options:

- `--include-single-agent`: also add a synthetic `single-agent` baseline row
  from `selected_runs`. By default, the script only scores the architectures
  present in `paired_metrics`.
- `--output-dir <dir>`: choose a different output directory.
- `--model <model>` and `--token-budget <budget>`: score a different model or
  token-budget slice, as long as the matching paper-metrics summaries exist.

The script prints the Mean Absolute Error (MAE) to the console and displays a
table sorted by predicted success rate, with `predicted_success_rate`,
`actual_success_rate`, and `absolute_error` for each architecture.

## Output Structure

Experiment outputs are saved to `exp_outputs/{dataset_id}/{agent}/{model}/{date}/{time}/`.

The BrowseComp+ selector is `dataset=browsecomp-plus`, and its current dataset id is `browsecomp_plus_sampled_100`, so those runs are written under `exp_outputs/browsecomp_plus_sampled_100/...`.

Example:

```
exp_outputs/
└── plancraft-test/
    └── multi-agent-centralized/
        └── gemini/
            └── gemini-2.0-flash/
                └── 2026-04-04/
                    └── 01-09-04/
                        ├── .hydra/
                        ├── run_config.yaml
                        ├── run.log
                        ├── run_experiment.log
                        ├── dataset_eval_metrics.json
                        ├── run_runtime_metrics.json
                        └── instance_runs/
                            └── 0000/
                                ├── instance_save.yaml
                                ├── runtime_metrics.json
                                ├── runtime_events.jsonl
                                └── *_output.yaml
```

### Output Files

- `run_config.yaml`: resolved experiment metadata, including the reference token budget and run limits
- `dataset_eval_metrics.json`: dataset-level evaluation summary
- `run_runtime_metrics.json`: aggregated runtime metrics across completed instances in the run
- `instance_runs/<idx>/runtime_metrics.json`: per-instance raw runtime metrics
- `instance_runs/<idx>/runtime_events.jsonl`: per-instance raw event trace
- `instance_runs/<idx>/instance_save.yaml`: input, output, and evaluation payload for that instance
- `instance_runs/<idx>/*_output.yaml`: architecture-specific saved agent output

## Example Output

See `example_outputs/` directory for sample experiment outputs demonstrating:
- Single-agent execution traces
- Multi-agent coordination logs
- Evaluation metrics

## Project Structure

```
agent-scaling/
├── agent_scaling/           # Main Python package
│   ├── agents/              # Agent implementations
│   │   ├── single_agent.py
│   │   ├── multiagent_centralized.py
│   │   └── ...
│   ├── datasets/            # Dataset loaders
│   ├── env/                 # Environment & tools
│   ├── llm/                 # LLM integration
│   └── config/              # Configuration classes
├── run_scripts/             # Entry points
│   └── run_experiment.py
├── run_conf/                # Hydra configurations
│   ├── agent/               # Agent configs
│   ├── dataset/             # Dataset configs
│   └── run_exp.yaml         # Master config
├── datasets/                # Dataset files
├── prompts/                 # Prompt templates
└── example_outputs/         # Sample outputs
```

## Configuration Reference

### Master Config (`run_conf/run_exp.yaml`)

```yaml
defaults:
  - agent: multi-agent-centralized  # Agent type
  - dataset: plancraft-test         # Dataset

llm:
  model: gemini/gemini-2.0-flash    # LLM model
  params:
    temperature: 0.0                # Generation temperature

log_langfuse: false                 # Enable LangFuse tracing
use_disk_cache: false               # Keep false for research comparisons; true reuses saved LLM responses
num_workers: 1                      # Parallel workers
debug: true                         # Debug mode
max_instances: 3                    # Max instances to process
token_budget:
  enabled: true
  total_tokens_per_instance: 4800   # Monitored reference budget; iteration caps remain the hard limit
```

### Multi-Agent Config (`run_conf/agent/multi-agent-centralized.yaml`)

```yaml
name: multi-agent-centralized
prompts:
  lead_agent:
    local_path: prompts/multi-agent/lead_agent.yaml
  subagent:
    local_path: prompts/multi-agent/subagent.yaml

agent_specific_config:
  n_base_agents: 3
  min_iterations_per_agent: 3
  max_iterations_per_agent: 3
  max_rounds: 5
  task_blurb: "task coordinator"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kim2025towards,
  title={Towards a science of scaling agent systems},
  author={Kim, Yubin and Gu, Ken and Park, Chanwoo and Park, Chunjong and Schmidgall, Samuel and Heydari, A Ali and Yan, Yao and Zhang, Zhihan and Zhuang, Yuchen and Malhotra, Mark and others},
  journal={arXiv preprint arXiv:2512.08296},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
