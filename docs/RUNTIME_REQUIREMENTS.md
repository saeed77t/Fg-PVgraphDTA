# Strict research runtime setup

Research CLI execution deliberately requires more manual setup than installation.
This requirement does not change model mathematics, preprocessing or datasets.
It does not provide author permission checks, licensing or resistance to code edits.

## Required method environments

Both method families require Python **3.10** and these exact release versions:

| Dependency | Version |
|---|---|
| NumPy | 1.26.4 |
| RDKit | 2024.3.6 |
| PyTorch | 2.4.1 |
| PyTorch Geometric | 2.6.1 |

The `pv` family additionally requires torchvision **0.19.1** and Pillow **10.4.0**.
CUDA/CPU local wheel tags are accepted when creating a profile, but the exact
installed build is recorded and must match on subsequent execution. These are
the repository's existing tested pins, now checked at runtime; this change did
not downgrade the environment or invent incompatible package combinations.
Other installation dependencies remain pinned in `pyproject.toml` and `uv.lock`.

## Required manual steps

1. Create a dedicated Python 3.10 environment and install the pinned dependencies.
2. Explicitly create a profile with `fgpvdta configure-runtime --methods fg pv
   --output .runtime/research.json`. Use only `fg` or `pv` to restrict the enabled
   method families. The profile is a local environment record, not a secret key.
3. Pass `--runtime-profile .runtime/research.json` to every `train`, `suite` and
   `predict` command, including calls through the Python/TypeScript launchers.
4. If the interpreter path, Python patch release, platform or recorded dependency
   versions change, create a new profile at a new path. No automatic regeneration
   or fallback occurs. Do not distribute your machine-bound profile on GitHub.

All requested families are checked before an all-suite training job starts.
An FG-only profile cannot execute a PV suite. An unchanged valid profile remains
usable for repeated runs. No expiration, network call or license server is involved.

## Separate structural prediction environment

HHblits still requires its executable and sequence database, selected with
`align --executable ... --database ...`. PconsC4 runs under an explicitly supplied
external legacy Python interpreter via `predict-contacts --python ...`; it is
not installed into the research environment. Its upstream software, weights and
compatible legacy dependencies must be supplied separately. Fresh structural
prediction has not been validated locally; existing structural assets were used
for prior real-data preprocessing checks.

## Scope and limitations

Only the research CLI commands above enforce this setup gate. Source inspection,
library calls, tests, synthetic smoke checks, `--plan`, preprocessing, parameter
counts and result analysis remain available without profiles. This makes the
supported research workflow more demanding to set up, not impossible to bypass.
Existing saved checkpoints/results are not modified by profile creation.
