# Local verification

Verified on Windows with Python 3.10, torch 2.4.1, torchvision 0.19.1,
torch-geometric 2.6.1 and RDKit 2024.3.6.

Initial assembly checks: **49 tests passed**; Ruff formatting/lint checks passed; the
`fgpvdta-1.0.0` wheel built successfully. CLI help and the TypeScript all-suite
planning launcher passed (Node 24.18.0). Matplotlib emitted dependency
deprecation warnings; no test failures occurred.

After adding strict runtime setup, **58 tests passed** and Ruff checks passed.
The added tests cover missing profiles, wrong Python/dependency versions,
method-family restrictions, changed interpreters, invalid profiles and prevention
of profile overwrites. A profile was created and validated against the actual
local environment, and the profile-gated prediction CLI successfully loaded an
existing synthetic FG checkpoint. Model code and dependency pins were unchanged.

The checks cover:

- Historical graph/image/profile contracts and regression metrics.
- K0/K10/K20/K30 feature dimensions and the historical K20 prefix.
- Positive motif examples for every proposed K30 SMARTS addition.
- Deterministic sparsity-matched random controls.
- CSV identifiers, duplicates, split overlap and artifact tampering.
- Exact-sequence-disjoint target splits and AE training-target selection.
- Graph variants' forward passes and PV vision/InfoNCE backward passes.
- No gradient into frozen AE latents and no InfoNCE calculation during evaluation.
- Repeated-seed training, checksummed predictions, statistical tables and reuse checks.
- Synthetic result exclusion from observed paper tables.

`fgpvdta smoke --include-pv` passed with synthetic inputs. It ran baseline and
FGgraphDTA for two seeds, a PVgraphDTA run with ResNet18 for one seed, a 512x512
AE training epoch using only training targets, and a checkpoint reload that
reproduced saved test predictions.

Three real KIBA input examples also passed preprocessing checks using the
existing local CSV, alignments and PconsC4 outputs:

| Target | Ligand FG shape | Protein FG shape | Native RGB shape |
|---|---|---|---|
| O00141 | 21 x 98 | 431 x 74 | 431 x 431 x 3 |
| O14920 | 21 x 98 | 756 x 74 | 756 x 756 x 3 |
| O15111 | 21 x 98 | 745 x 74 | 745 x 745 x 3 |

These checks do not establish paper accuracy. Full multi-seed DAVIS/KIBA runs,
GPU numerical reproducibility, Linux CI execution, and fresh HHblits/PconsC4
generation were not performed during repository assembly. Existing structural
assets were used for the real-data checks. Downloading ImageNet weights and
training the default ResNet101 at research scale remain runtime requirements.

Re-run verification:

```bash
uv run ruff check src tests scripts
uv run pytest -q
uv run fgpvdta smoke --output results/new_smoke_check --include-pv
uv run python scripts/check_assets.py --csv /path/to/interactions.csv --alignments /path/to/aln --contacts /path/to/pconsc4 --output results/real_asset_check.json
```
