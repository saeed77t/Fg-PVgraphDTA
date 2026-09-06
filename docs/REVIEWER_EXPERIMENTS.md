# Reviewer experiment definitions

All executable variant definitions live in `experiments/presets.py`; all suites
use `experiments/runner.py`. Dataset choice is the prepared directory, so DAVIS
and KIBA use the same code and separate output roots.

## FG studies

Baseline uses 78 ligand / 54 protein inputs. Ligand-only adds 20 ligand bits;
protein-only adds 20 protein bits; bilateral FG adds both. FG20 is historical.

K10 is defined here as the first ten patterns in historical order; this is an
explicit reproducible subset, not a claim that it is the optimal ten. K30 retains
all historical twenty and appends the ten motifs in `feature_sets.py`:
organic halide, alkene, alkyne, imine, disulfide, sulfonamide, sulfonic acid,
phosphate, phosphonate and aromatic heteroatom. These additional SMARTS are a
**new proposed protocol**, absent from the supplied notebooks and manuscript.
Review their definitions before using K30 in a manuscript response. No prior K30
performance or reviewer-approved vocabulary is claimed. Preparation serializes
the exact names and patterns into its manifest.

Random control independently permutes each node's 20 FG bits, preserving each
node's active-bit count. A stable hash of modality/entity ID and control seed
fixes that randomization across training seeds and does not use labels. This
controls feature dimension and row sparsity, but retains the original count of
active chemical motifs; it is not a fully chemistry-free Bernoulli control.

The original GCN hidden widths scale with input feature dimensions. Consequently
K10/K20/K30 also change parameter counts. The counts are saved and must accompany
any interpretation; these are not parameter-matched capacity controls.

## PV studies

Graph-only -> graph+vision -> graph+vision+frozen AE gate -> full model with
three symmetric InfoNCE pairs. The loss is MSE + alpha * InfoNCE with historical
alpha=0.5, tau=0.07. The sensitivity suite varies one axis at a time; its central
configuration appears only once. ResNet18/101 comparison is separate. Vision
models use base 78/54 graph features, as in the final PV notebooks.

## Cold-target studies

`cold_target` compares baseline, FGgraphDTA and PVgraphDTA. Any other suite can
also be made cold-target with `--cold-target`. Split by exact sequence, keeping
duplicate sequence IDs together. The default target-group fractions are
80/10/10, with group rounding recorded by actual persisted row lists. To use
20% held-out targets while retaining validation, choose `--test-fraction 0.2
--validation-fraction 0.1` (70/10/20 target groups).

Protein counts are derived from actual inputs. No hardcoded 229/58 split or
literature performance is inserted. The AE is fitted on training targets for
each split and then frozen. Test affinities are evaluated after selecting the
checkpoint on validation MSE. Similar proteins can still lie in different
partitions; sequence-similarity clusters are a distinct protocol.

## Results

Five default training seeds share one split; they measure initialization/training
variability, not variability over random data partitions. To study split
variability, use different `--split-seed` values and separate output roots.
Report those split replicates separately rather than pooling them as matched
training seeds. Statistical code rejects mismatched splits and missing seeds.

All comparisons are added code, not evidence of completed scientific experiments.
The synthetic smoke run is explicitly excluded from observed-result tables.
