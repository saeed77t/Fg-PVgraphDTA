# FGgraphDTA and PVgraphDTA

Complete research source for both methods: raw-data preparation, molecular and
protein graphs, functional groups, contact images, autoencoder fitting, model
training, ablations, cold-target evaluation, prediction, and statistical analysis.
There are no notebooks or omitted/private model components in this repository.
Scientific code is Python; `scripts/run_all.ts` is an optional TypeScript launcher.

## Install and check

Use Python 3.10 (the tested version) and run from this repository's root:

```bash
uv sync --extra dev --python 3.10
uv run fgpvdta --help
uv run pytest -q
uv run fgpvdta smoke --output results/smoke --include-pv
```

The smoke command builds synthetic graphs and images, trains the graph methods,
fits an AE on training targets, trains a ResNet18 PV model, and verifies a
checkpoint reload. Its output is marked synthetic and excluded from paper tables.
The research PV default is ResNet101 with ImageNet initialization.

Without uv, create/activate a Python 3.10 virtual environment and run
`python -m pip install -e ".[dev]"`; then use `fgpvdta` directly.

## Mandatory manual setup for research execution

Installing the package is no longer sufficient to run `train`, `suite`, or
`predict`. Those CLI commands require a local runtime profile and enforce
Python 3.10 plus the pinned method dependencies on each invocation:

```bash
uv run fgpvdta configure-runtime --methods fg pv --output .runtime/research.json
```

Pass `--runtime-profile .runtime/research.json` explicitly to each research
command. `fg` enables the graph baseline and FG variants; `pv` enables the PV
family, including its graph-only ablation. A profile is bound to its interpreter
path, Python patch version, platform and package versions. Moving/recreating an
environment requires generating a new profile; existing files are never replaced
automatically. `.runtime/` is excluded from Git.

These are intentional setup restrictions, **not licensing or security controls**.
The complete source and library APIs remain available. Preprocessing, statistics,
`--plan`, parameter counting and synthetic smoke tests do not require a profile.
See [RUNTIME_REQUIREMENTS.md](docs/RUNTIME_REQUIREMENTS.md) for exact restrictions.

## Repository map

| Component | Code |
|---|---|
| Dataset conversion, validation, splits | `src/fgpvdta/data/` |
| Complete preprocessing workflow | `src/fgpvdta/preprocessing/pipeline.py` |
| FASTA, HHblits and PconsC4 wrappers | `src/fgpvdta/preprocessing/external.py` |
| Historical FG definitions | `src/fgpvdta/preprocessing/functional_groups.py` |
| K10/K20/K30 and random controls | `src/fgpvdta/preprocessing/feature_sets.py` |
| FGgraphDTA | `src/fgpvdta/models/fggraphdta.py` |
| PVgraphDTA, fusion and InfoNCE | `src/fgpvdta/models/pvgraphdta.py` |
| Contact-map autoencoder | `src/fgpvdta/models/autoencoder.py` |
| Training, checkpoints and prediction | `src/fgpvdta/training/` |
| All experiment configurations | `src/fgpvdta/experiments/presets.py` |
| Repeated-seed and cold-target runner | `src/fgpvdta/experiments/runner.py` |
| Tables and paired statistical tests | `src/fgpvdta/analysis/reviewer.py` |

## 1. Prepare the input data

If you already have an affinity CSV, it must have these columns:

```text
Drug_ID,Drug,Target_ID,Target,Y
```

`Drug` is SMILES, `Target` is the amino-acid sequence, and `Y` is the model's
affinity label (DAVIS pKd or KIBA score). Existing CSV labels are not transformed
again. IDs must be consistent, safe filenames; duplicate pairs and inconsistent
ID-to-sequence/SMILES mappings are rejected.

Alternatively, download and convert the original DeepDTA format:

```bash
uv run fgpvdta download --output data/raw/deepdta
uv run fgpvdta canonicalize --dataset davis --input data/raw/deepdta/davis --output data/davis.csv
uv run fgpvdta canonicalize --dataset kiba --input data/raw/deepdta/kiba --output data/kiba.csv
```

For an archived study, supply an upstream commit SHA with `download --revision`.
The downloader records the source archive hash. Only deserialize trusted data
archives and graph/checkpoint files; the historical formats use pickle.

## 2. Alignments and contact maps

Supply one `<Target_ID>.aln` alignment and one `<Target_ID>.npy` contact matrix for
each target. Existing assets can be used directly if IDs and sequence lengths
match the CSV. Contact matrices must contain probabilities in [0,1].

To generate them from sequences:

```bash
uv run fgpvdta export-fasta --csv data/davis.csv --output data/davis/fasta
uv run fgpvdta align --fasta data/davis/fasta --database /path/to/hhblits/database --output data/davis/alignments
uv run fgpvdta predict-contacts --alignments data/davis/alignments --python /path/to/pconsc4/python --output data/davis/contacts
```

HHblits, its database, and the historical PconsC4 environment are external
requirements. Their wrappers are included; databases and prediction weights are
not bundled. See [PREPROCESSING.md](docs/PREPROCESSING.md) for contracts and provenance.

## 3. Build graphs and images

```bash
uv run fgpvdta prepare --csv data/davis.csv --alignments data/davis/alignments --contacts data/davis/contacts --output data/prepared/davis
uv run fgpvdta verify --data data/prepared/davis
```

This creates all baseline/FG graph variants and images together, using the same
interaction population. It writes source/artifact checksums, the exact FG
definitions, and a filtering report. Missing structural assets stop preparation;
`--allow-missing` explicitly opts into a reported structural subset.

Output directories must be empty. Use a new directory when changing inputs.
The historical KIBA pair-selection option is `prepare --legacy-kiba-subset`:
first/third/fifth structurally valid rows, capped at 40,000. Full-data processing
is the default. This option does not reproduce the old notebook's test-set reuse.

## 4. Main methods and all reviewer experiments

```bash
uv run fgpvdta train --model fggraphdta --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
uv run fgpvdta train --model pvgraphdta --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
uv run fgpvdta suite --name all --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
```

Repeat with `data/prepared/kiba` and `results/kiba` for KIBA. All runs share a
persisted split for the same dataset/protocol. Defaults are seeds 11/22/33/44/55,
80/10/10 partitions, 250 maximum epochs, validation-based checkpoint selection,
Adam 0.001, and dropout 0.2. Options are listed by `fgpvdta suite --help`.

| Suite (`--name`) | Variants |
|---|---|
| `methods` | DGraphDTA-style baseline, FGgraphDTA, PVgraphDTA |
| `fg_ablation` | Baseline, ligand-only FG, protein-only FG, bilateral FG |
| `fg_count` | K10, historical K20, proposed extended K30 |
| `random_control` | Baseline, real FG, random FG control |
| `pv_ablation` | Graph-only, graph+vision, graph+vision+AE, full InfoNCE model |
| `sensitivity` | Alpha 0/0.1/0.5/1.0; tau 0.03/0.07/0.1/0.2, one axis at a time |
| `backbone` | ResNet18 versus ResNet101 |
| `cold_target` | All three main models on protein-disjoint splits |

K10/K30 and the random-control protocol are explicit new experiment definitions;
they are not claimed to be historical results. See [REVIEWER_EXPERIMENTS.md](docs/REVIEWER_EXPERIMENTS.md).

For cold-target main runs or **all ablations on cold-target splits**:

```bash
uv run fgpvdta suite --name cold_target --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
uv run fgpvdta suite --name all --cold-target --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
```

Cold partitions group identical protein sequences together even if their IDs
differ. Each split's AE is fitted only on training-target images, then frozen
before latent export and DTA training. AE artifacts are reused only when their
split, fitting configuration and input hashes match. Test labels are never used
for early stopping. Similarity-cluster-disjoint splitting is not implied.

Use `--plan` to list configurations without running or loading data:

```bash
uv run fgpvdta suite --name all --data data/prepared/davis --plan
uv run python scripts/run_all.py --data data/prepared/davis --output results/davis --runtime-profile .runtime/research.json
```

Optional TypeScript launcher (Node with TypeScript stripping support): set
`PYTHON_EXECUTABLE` to this environment's Python, then run:

```bash
node --experimental-strip-types scripts/run_all.ts --data data/prepared/davis --plan
```

## 5. Predictions, statistical tests and tables

Each run saves `configuration.json`, `checkpoints/best.pt`, `last.pt`, epoch
history, test predictions, per-target metrics, and a checksummed `result.json`.
Completed identical runs are skipped after verification. Changed configurations
or incomplete runs require a new output directory; they are never silently
overwritten. Use the saved checkpoint for test-set prediction:

```bash
uv run fgpvdta predict --run results/davis/methods/fggraphdta/seed_11 --data data/prepared/davis --output results/fg_reloaded.csv --runtime-profile .runtime/research.json
uv run fgpvdta parameters --suite methods --output results/model_parameters.json
uv run fgpvdta summarize --results results/davis/fg_ablation --output results/davis/tables/fg --baseline dgraphdta
uv run fgpvdta summarize --results results/davis/cold_target --output results/davis/tables/cold --baseline dgraphdta
uv run fgpvdta summarize --results results/davis/pv_ablation --output results/davis/tables/pv --baseline graph_only
```

Tables recompute metrics from saved predictions and require matching seeds and
split hashes. They include mean/sample SD, paired seed tests, paired target-MSE
tests after averaging over seeds, bootstrap 95% intervals, Holm-adjusted p-values,
parameter counts, runtime and GPU memory. Positive effect means improvement.
See [STATISTICS.md](docs/STATISTICS.md) for units of analysis and limitations.

## Provenance and verification scope

The source audit used `codes all i have !`, `V4.1-Paper (1).docx`, and the
"paper revised / Generate Ablation Codes" conversation. Notebook hashes and
definition indexes are in [source_inventory.json](docs/source_inventory.json);
notebook files themselves are excluded. Original and new behavior are separated
in [HISTORICAL_DIFFERENCES.md](docs/HISTORICAL_DIFFERENCES.md).

This repository supplies executable experiment code. Full DAVIS/KIBA repeated
training results are **not** precomputed or claimed. See [VALIDATION.md](docs/VALIDATION.md)
for exactly what was tested locally. Trained weights, databases, datasets and
generated results are ignored by Git. This complete-source folder should be
kept private if access to the method implementation is intended to remain restricted.
