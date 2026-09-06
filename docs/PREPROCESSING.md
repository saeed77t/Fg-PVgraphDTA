# Preprocessing contracts

The executable order is CSV conversion -> FASTA -> HHblits/A3M -> ALN -> PconsC4
contact probabilities -> graph/image preparation -> persisted split -> training-only
AE fitting -> frozen latent export -> DTA model training -> test evaluation.

The `.aln` files are plain aligned sequences, one per line. The A3M converter
removes lowercase insertions and dot insertions, retains gaps, and checks aligned
lengths. `external.py` invokes HHblits using an argument list (no shell expansion)
and PconsC4 in a separate Python environment for each target. Its API calls are
recovered from `#2 Create Contact Maps.ipynb`. These external executables, databases
and legacy prediction dependencies must be installed separately.

## Ligand graphs

The historical 78 features are 44 atom symbols, 11 degree flags, 11 total-H flags,
11 implicit-valence flags, and aromaticity. Edges represent bonds in both
directions. FG20 adds the exact original per-atom SMARTS multi-hot bits; its first
20 columns are preserved in all extended feature sets. Atom features are not
silently normalized or standardized. `smiles.py` exposes explicit alternative
canonical/fragment-parent policies, while the main pipeline uses notebook parsing.

## Protein graphs

Base features contain 21 residue-identity bits, 21 alignment-profile values,
five residue class flags and seven historical physicochemical values (54 total).
The source calls its profile a PSSM, but it is a pseudocount-smoothed frequency
profile, not a log-odds matrix. The historical denominator counts all nonempty
alignment lines; only length-matched lines contribute residue counts. The
`strict` library option uses length-matched lines for the denominator.

Protein FG20 is computed using the exact supplied free-amino-acid SMILES and
overlap cleanup. These strings and redundant backbone groups are preserved as
historical behavior; they are not claimed to be corrected residue chemistry.
The main pipeline uses the upper triangle at contact probability >=0.5 and adds
both edge directions. GCNConv supplies its own self-loops. For dense input,
contact values and the image diagonal are preserved; no symmetrization is applied
by the main pipeline. Sparse text formats necessarily reconstruct symmetric maps.

## Images and autoencoder

Native LxL RGB: red is min-max contact probability; green is the min-max mean of
row/column sinusoidal positional encodings and 21-AA one-hot planes; blue is zero.
The feature-plane mean is calculated algebraically to avoid allocating 106 LxL
planes. Floating-point summation can differ at the last bits. PNG uses the original
`origin=lower`. PV preparation resizes to 256 with Pillow's omitted-resample default,
then the loader resizes to 224 and applies ImageNet normalization. AE training uses
512x512 RGB, five convolution stages, a 128-D latent, and masked MSE at 0.05.
The final decoder reconstructs red/green and appends zero blue.

AE training happens after splitting. Only training-target images enter the
optimizer. Inference then exports latents for all targets using the frozen encoder.
Missing images, wrong-size latents and changed checksums fail explicitly.

## Prepared directory

```text
prepared/
  interactions.csv
  manifest.json
  filter_report.json
  ligands/{k0,k10,k20,k30,random_k20}/<Drug_ID>_graph.pt
  proteins/{k0,k10,k20,k30,random_k20}/<Target_ID>_graph.pt
  images/native/<Target_ID>_stacked.png
  images/resize256/<Target_ID>_stacked.png
```

All models compare the same prepared interaction population. Filtering is explicit
and reported; failed parsing or mismatched lengths never become fabricated data.
The initial source checksums and all prepared-file hashes are recorded. Use new
output directories for changed data/protocols.
