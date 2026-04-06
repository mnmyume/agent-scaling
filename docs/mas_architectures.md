# MAS Architectures

This repository now supports the four multi-agent system (MAS) topologies used in "Towards a Science of Scaling Agent Systems" for the currently implemented benchmarks and environments:

- `multi-agent-independent`
- `multi-agent-centralized`
- `multi-agent-decentralized`
- `multi-agent-hybrid`

Finance-Agent and Workbench are intentionally deferred in this pass.

## Files Added Or Introduced For This Work

- `agent_scaling/agents/multiagent_independent.py`
- `agent_scaling/agents/multiagent_decentralized.py`
- `agent_scaling/agents/multiagent_hybrid.py`
- `agent_scaling/agents/multiagent_components/system.py`
- `agent_scaling/agents/multiagent_components/budgeting.py`
- `run_conf/agent/multi-agent-independent.yaml`

## Key Supporting Updates

- `agent_scaling/agents/multiagent_centralized.py`
- `agent_scaling/agents/multiagent_components/conversation.py`
- `agent_scaling/agents/multiagent_components/mas_lead_agent.py`
- `agent_scaling/agents/multiagent_components/mas_subagent.py`
- `agent_scaling/agents/multiagent_components/memory.py`
- `agent_scaling/agents/__init__.py`
- `agent_scaling/config/run.py`
- `prompts/multi-agent/subagent.yaml`
- `run_conf/agent/multi-agent-centralized.yaml`
- `run_conf/agent/multi-agent-decentralized.yaml`
- `run_conf/agent/multi-agent-hybrid.yaml`
- `agent_scaling/env/browsecomp.py`
- `agent_scaling/utils/token_budget.py`
- `agent_scaling/utils/__init__.py`
- `agent_scaling/utils/core.py`

## Topology Mapping

### Independent

- Multiple workers run in parallel from a shared task instance.
- There is no orchestrator guidance during execution.
- There is no peer-to-peer exchange.
- Workers only contribute to final aggregation after their local work is complete.

Implementation:
- `agent_scaling/agents/multiagent_independent.py`

### Centralized

- A lead agent plans, coordinates, and decides when to stop.
- Workers only communicate with the orchestrator.
- This matches the orchestrator-to-worker star topology from the paper.

Implementation:
- `agent_scaling/agents/multiagent_centralized.py`
- `agent_scaling/agents/multiagent_components/mas_lead_agent.py`

### Decentralized

- There is no orchestrator.
- Workers execute explicit peer communication rounds.
- Finalization is consensus-based using worker final-answer proposals.

Implementation:
- `agent_scaling/agents/multiagent_decentralized.py`

### Hybrid

- A lead agent remains the main control point.
- Workers also participate in bounded peer exchange rounds.
- Peer communication is explicit, limited by config, and does not replace orchestration.

Implementation:
- `agent_scaling/agents/multiagent_hybrid.py`

## Budget Allocation

All MAS variants share one hard per-instance token budget through `TokenBudgetManager`. This preserves comparability with the single-agent setup and prevents silent budget bypass.

On top of the shared hard cap, `MASBudgetAllocator` adds topology-aware soft allocation buckets:

- `planning`
- `coordination`
- `worker`
- `peer`
- `synthesis`
- `consensus`

The allocator:

- starts from the same total instance budget used by SAS
- estimates expected calls from the selected topology
- derives per-bucket soft allocations
- caps per-call completion tokens
- still enforces the same global hard stop through the shared budget manager

This means MAS can redistribute a fixed total budget across agents, rounds, and synthesis without silently exceeding the matched overall budget.

## Example Commands

All commands below use the existing `python run_scripts/run_experiment.py ...` interface after activating `.venv`. If you prefer, prepend `uv run`.

PlanCraft:

```bash
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test
python run_scripts/run_experiment.py agent=multi-agent-independent dataset=plancraft-test
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=plancraft-test
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=plancraft-test
```

BrowseComp+:

```bash
python run_scripts/run_experiment.py agent=single-agent dataset=browsecomp-plus
python run_scripts/run_experiment.py agent=multi-agent-independent dataset=browsecomp-plus
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=browsecomp-plus
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=browsecomp-plus
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=browsecomp-plus
```

Tiny smoke runs:

```bash
python run_scripts/run_experiment.py agent=single-agent dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-independent dataset=plancraft-test debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=browsecomp-plus debug=true max_instances=1
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=browsecomp-plus debug=true max_instances=1
```

Useful overrides:

```bash
python run_scripts/run_experiment.py agent=multi-agent-decentralized dataset=plancraft-test max_instances=1 debug=true
python run_scripts/run_experiment.py agent=multi-agent-hybrid dataset=browsecomp-plus token_budget.total_tokens_per_instance=4800
```

Metrics aggregation:

```bash
uv run python run_scripts/aggregate_metrics.py exp_outputs/plancraft-test
uv run python run_scripts/aggregate_metrics.py exp_outputs/browsecomp_plus_sampled_100
uv run python run_scripts/materialize_paper_metrics.py exp_outputs/plancraft-test
```
