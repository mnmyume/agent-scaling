# Example Outputs

This directory contains sample experiment configurations, results, and execution traces demonstrating the framework's capabilities.

## Directory Structure

```
example_outputs/
├── plancraft-test/
│   └── multi-agent-centralized/    # Multi-agent on PlanCraft task
│       ├── run_config.yaml         # Experiment configuration
│       ├── run_trace.txt           # Execution trace with prompts and responses
│       ├── dataset_eval_metrics.json  # Dataset-level eval results
│       └── run_runtime_metrics.json   # Run-level runtime metrics
└── browsecomp-plus/
    ├── single-agent/               # Single agent on BrowseComp+ task
    │   ├── run_config.yaml
    │   └── dataset_eval_metrics.json
    └── multi-agent-centralized/    # Multi-agent on BrowseComp+ task
        ├── run_config.yaml
        └── dataset_eval_metrics.json
```

## Output Files

### `run_config.yaml`
Complete configuration used for the experiment, including agent type, dataset, LLM model, and all hyperparameters.

### `run_trace.txt`
Detailed execution trace including:
- System and user prompts sent to the LLM
- LLM responses with token counts
- Tool calls and their results
- Agent coordination messages (for multi-agent systems)

### `dataset_eval_metrics.json`
Aggregated evaluation metrics:
```json
{
  "avg_success": 0.82,      // Average task success rate (PlanCraft)
  "avg_accuracy": 0.60,     // Average accuracy (BrowseComp+)
  "num_instances": 100      // Number of instances evaluated
}
```

### `run_runtime_metrics.json`
Aggregated runtime metrics written by the new trace collector:
```json
{
  "architecture": "multi-agent-centralized",
  "model": "minimax/MiniMax-M2.7",
  "token_budget": 4800,
  "instance_count": 100,
  "total_turns": 742.0,
  "total_messages": 318.0,
  "message_density_c": 0.4286,
  "total_tokens": 278400.0,
  "total_llm_calls": 742.0,
  "total_tool_calls": 514.0,
  "success_rate": 0.82,
  "success_per_1k_tokens": 0.2945
}
```

## Reproducing These Results

### PlanCraft Multi-Agent
```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash
```

### MiniMax M2.7 Preset
```bash
python run_scripts/run_experiment.py --config-name run_exp_minimax
```

Override agent or dataset as usual:
```bash
python run_scripts/run_experiment.py --config-name run_exp_minimax agent=single-agent
python run_scripts/run_experiment.py --config-name run_exp_minimax dataset=browsecomp-plus
```

### BrowseComp+ Single-Agent
```bash
python run_scripts/run_experiment.py \
    agent=single-agent \
    dataset=browsecomp-plus \
    llm.model=openai/gpt-5-mini
```

### BrowseComp+ Multi-Agent
```bash
python run_scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=browsecomp-plus \
    llm.model=openai/gpt-5
```

Note: Exact results may vary due to LLM non-determinism even with temperature=0.
