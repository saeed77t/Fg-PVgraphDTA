# Source mapping

Sources were read from the parent's `codes all i have !` directory. The full
inventory records each notebook's SHA-256, code-cell count and parsed definition
names. Original cells were not copied wholesale or executed as notebooks.

| Implementation | Original source |
|---|---|
| Ligand base features | baseline-molecular-graphs.ipynb |
| Ligand SMARTS FG20 | functional-group-molecular-graphs.ipynb |
| Residue features and profile | baseline-protein-graphs.ipynb |
| Protein FG20 and free-AA strings | functional-group-protein-graphs.ipynb |
| FGgraphDTA graph architecture | kiba-fggraphdta.ipynb; kiba-fggragpdta-final.ipynb |
| PV architecture and three-pair InfoNCE | kiba-pvgraphdta.ipynb; kiba-gnn-cnn-infonce.ipynb; davis-pvgraphdta.ipynb |
| RGB image contract | contact-map-image-construction.ipynb; onehot+PE.ipynb |
| AE encoder and RG decoder | kiba-autoencoder.ipynb; davis-autoencoder-variants.ipynb |
| PconsC4 process wrapper | #2 Create Contact Maps.ipynb |

`V4.1-Paper (1).docx` was inspected as the manuscript reference; it states 80/10/10
train/validation/test partitions. It was not copied into the code repository.
The final runner implements that protocol and distinguishes historical differences.

The Chrome project conversation "paper revised / Generate Ablation Codes" was
read, and a focused audit was requested there on 2026-09-06. It identified FG
ablation, K10/K30, random-FG, cold-target, per-target tests and alpha/tau sweeps
as reviewer additions rather than completed notebook experiments. The local
implementation was checked against the notebooks, including the 20 SMARTS,
PV graph dimensions, AE layers, InfoNCE summation and KIBA valid-row counter.

The earlier modular reconstruction supplied the initial package layout and
tested primitives. This final repository consolidates them under one `fgpvdta`
namespace and adds a shared experiment runner, strict data/seed checks, split-aware
AE fitting, K variants, random control, target-level tests and integrated commands.
Checkpoint parameter names have been refactored; old notebook checkpoints are
not claimed to load unchanged. This is a reimplementation, not bitwise historical
result reproduction. No notebook result numbers were copied into new outputs.
