# Obsidian Templates — ML Zettelkasten

## What is this?

A set of Zettelkasten templates tailored for machine learning research, study, and experimentation. Each template enforces atomic note-taking while capturing the specific structure ML work demands.

## Included templates

| Template | Type | When to use |
|----------|------|-------------|
| `Fleeting Note.md` | Quick capture | An ML idea, intuition, or observation you want to remember. |
| `Paper Note.md` | Literature note | Summarizing an ML paper, blog post, or technical article. |
| `Concept Note.md` | Permanent note | An atomic ML/stats concept (e.g., attention, regularization, KL divergence). |
| `Model Card.md` | Permanent note | A model's architecture, tradeoffs, complexity, and when to use it. |
| `Experiment Log.md` | Project note | Tracking a specific experiment: hypothesis, setup, results, next steps. |
| `Research Log.md` | Daily note | Daily research journal: reading, experiments, ideas, debugging. |
| `Research Question.md` | Question note | An open ML question you're investigating. |
| `MOC.md` | Map of Content | A thematic index grouping related ML notes (e.g., "Transformers", "Time Series"). |

## Workflow

```
┌────────────────┐     ┌───────────────┐     ┌────────────────┐
│ Fleeting Note  │────▶│  Paper Note   │────▶│ Concept Note / │
│ (quick idea)   │     │ (read paper)  │     │ Model Card     │
└────────────────┘     └───────────────┘     └───────┬────────┘
                                                      │
         ┌───────────────────┐                        ▼
         │  Experiment Log   │◀──────────────┌────────────────┐
         │ (test hypothesis) │               │  MOC / Index   │
         └───────────────────┘               └────────────────┘
```

1. **Capture** → `Fleeting Note` for quick ideas while coding, reading, or in meetings.
2. **Read** → `Paper Note` to process papers systematically (problem → method → results → takeaways).
3. **Distill** → `Concept Note` for foundational ideas, `Model Card` for specific architectures.
4. **Experiment** → `Experiment Log` to document hypothesis-driven experiments with reproducible details.
5. **Connect** → Link everything. Use `MOC` to create navigable topic hubs.
6. **Reflect** → `Research Log` daily and `Research Question` for things to investigate.

## Note maturity

| Status      | Meaning                     |
| ----------- | --------------------------- |
| `inbox`     | Unprocessed capture         |
| `seedling`  | Initial draft, needs work   |
| `growing`   | Being refined and connected |
| `evergreen` | Mature, stable, well-linked |
| `open`      | Unanswered question         |
| `resolved`  | Question answered           |
| `active`    | Ongoing experiment/project  |
| `completed` | Finished experiment         |

## Obsidian setup

### Option 1: "Templates" plugin (core)

1. Copy this folder into your vault.
2. **Settings → Core plugins → Templates** → enable.
3. Set **Template folder location** to `obsidian_templates_ml`.
4. Use: `Ctrl/Cmd + P` → "Insert template" → pick one.

### Option 2: "Templater" plugin (community, recommended)

1. Install **Templater** from Community Plugins.
2. **Settings → Templater → Template folder** → `obsidian_templates_ml`.
3. Use: `Alt + N` → select template.

### Daily Research Log

1. **Settings → Core plugins → Daily notes** → enable.
2. Set **Template file location** to `obsidian_templates_ml/Research Log`.
3. A research log will be created automatically each day.

## Template variables

| Variable | Replaced by |
|----------|-------------|
| `{{title}}` | File/note name |
| `{{date:YYYY-MM-DD}}` | Current date |
| `{{date:YYYYMMDDHHmm}}` | Timestamp as unique ID |
| `{{date:YYYY-MM-DD, dddd}}` | Date with day of week |
| `{{date:YYYY-MM-DD HH:mm}}` | Date with time |

## Suggested MOC structure for ML

A good starting point for your ML knowledge base:

```
MOC: Machine Learning
├── MOC: Supervised Learning
│   ├── MOC: Linear Models
│   ├── MOC: Tree-Based Methods
│   ├── MOC: Neural Networks
│   │   ├── MOC: Transformers
│   │   ├── MOC: CNNs
│   │   └── MOC: RNNs
│   └── MOC: Evaluation & Metrics
├── MOC: Unsupervised Learning
├── MOC: Optimization
├── MOC: Probability & Statistics
└── MOC: MLOps & Production
```

## Conventions

- **Name notes after the insight**: "Batch normalization reduces internal covariate shift" > "Batch Norm"
- **Link aggressively**: every concept mention should be a `[[wikilink]]`.
- **One note, one idea**: split multi-concept notes and link them.
- **Include code**: a concept without a code snippet is incomplete for ML work.
- **Reproducibility matters**: experiment logs should have enough detail to re-run.

## Useful Dataview queries

### Seedling concepts to develop

```dataview
TABLE status, area, file.tags as tags
FROM ""
WHERE type = "concept" AND status = "seedling"
SORT created DESC
```

### Active experiments

```dataview
TABLE dataset, framework, status
FROM ""
WHERE type = "experiment" AND status = "active"
SORT created DESC
```

### Recently read papers

```dataview
TABLE authors, year, venue
FROM ""
WHERE type = "paper"
SORT created DESC
LIMIT 10
```
