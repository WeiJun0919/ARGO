# ARGO

**Adaptive and Fine-Grained Data Governance via Hierarchical Reinforcement Learning**

ARGO is a research implementation of an adaptive data-governance framework for
machine-learning training data. Instead of applying a fixed cleaning pipeline,
ARGO repeatedly observes the evolving dataset and downstream model, selects a
governance operation, and applies that operation to a bounded subset of candidate
samples.

The framework supports four heterogeneous operations:

- **Remove** potentially harmful samples.
- **Relabel** suspicious labels with a learned categorical correction head.
- **Repair** corrupted tabular features with a learned numerical correction head.
- **Augment** underrepresented or uncertain regions of the training distribution.

A **No-op** action allows the policy to abstain when no intervention is expected
to be beneficial.

## Overview

ARGO decomposes each governance decision into two levels:

1. A high-level PPO actor-critic selects the next governance operation.
2. An operation-specific low-level selector ranks candidate samples and determines
   the concrete modification.

The governance state combines global dataset/model statistics, operation-specific
signals, recent decision history, remaining budget, and temporal changes. A
learnable gating network reweights the four operation-specific state groups and is
optimized jointly with the PPO actor and critic.

```text
Dataset + downstream model
          |
          v
Governance-aware state encoder
          |
          v
Adaptive gate + high-level PPO
          |
          v
Operation-specific selector
          |
          v
Bounded data modification
          |
          v
Proxy or exact downstream evaluation
          |
          +--------------------------> next governance state
```

## Three-stage Training

ARGO uses a progressive training strategy to reduce hierarchical cold-start and
selector imbalance:

- **Stage I - Selector warm-up:** identify high-confidence samples, inject
  controlled label/feature corruptions, and initialize selection and correction
  heads with supervised objectives. The default is 6 epochs.
- **Stage II - Fixed-schedule refinement:** invoke operation-specific in strict
  round-robin order while the high-level PPO policy and gate are frozen. The
  default is 3 rounds per operation.
- **Stage III - Joint optimization:** jointly refine high-level scheduling,
  adaptive gating, and the invoked low-level selector using downstream utility.
  The default horizon is 100 governance decisions.

Each governance action processes at most 4% of the current training set by
default.

## Efficient Downstream Evaluation

The optional proxy evaluator models post-operation validation performance as a
Gaussian distribution conditioned on the current state, selected operation, and
a summary of the actual dataset change:

```text
(state, operation, rho(delta_dataset)) -> (mean performance, uncertainty)
```

Low-uncertainty predictions can be combined with differential downstream-model
updates. High uncertainty, periodic checkpoints, or a large accumulated data
change triggers exact validation and shared full retraining.

## Repository Structure

```text
argo/
├── data/                 Dataset containers, registry, diagnostics, and noise tools
├── governance/           Three-stage trainer, rewards, operations, and feedback
│   └── proxy_reward/     Uncertainty-aware proxy model and replay buffer
├── models/               State encoder and downstream models
├── rl/                   Governance environment and high-level PPO agent
└── selectors/            Operation-specific selectors, ILDE, and ED2
```

Important entry points:

- `argo/governance/trainer.py` - Stage I/II/III orchestration.
- `argo/rl/env.py` - one complete governance transition.
- `argo/rl/ppo_agent.py` - PPO actor, critic, and adaptive gate.
- `argo/models/state_encoder.py` - governance-state construction.
- `argo/selectors/neural_selector.py` - bounded target selection and correction.
- `argo/governance/proxy_reward/trainer.py` - proxy training and calibration.

## Requirements

- Python 3.8 or newer
- NumPy
- pandas
- scikit-learn
- PyTorch
- torchvision for image experiments

Install the base dependencies with:

```bash
python3 -m pip install numpy pandas scikit-learn torch
```

For image datasets, also install torchvision using the build compatible with
your PyTorch and CUDA versions.

## Reproducibility Notes

- Repair is available only for tabular data and is masked for image/text tasks.
- The environment supports rollback when an intervention reduces validation
  performance beyond the configured tolerance.
- Exact evaluations periodically recalibrate both the downstream model and proxy.
- Random seeds default to 42 in the main policy, models, and synthetic-data tools.

## Paper

This repository implements the method described in:

> Jun Wei, Shuang Hao, and Yingqi Mu. *Adaptive and Fine-Grained Data Governance
> via Hierarchical Reinforcement Learning*. 2026.

Please cite the paper if you use ARGO in academic work.
