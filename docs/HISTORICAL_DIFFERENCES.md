# Reproducibility Notes

## Split mismatch
The manuscript states 80/10/10. The supplied final FG/PV training notebooks use
80/20 without an explicit validation set and evaluate the test set each epoch.
This final repository uses 80/10/10 with validation-based checkpoint selection
and final held-out test evaluation. It does not claim identical historical scores.

## KIBA population
The manuscript describes 118,254 KIBA interactions. The historical final KIBA
notebooks retain the first/third/fifth structurally valid pairs up to 40,000.
The final pipeline processes all available pairs by default; explicit
`prepare --legacy-kiba-subset` reproduces that pair selection after structural
filtering. The new training protocol still retains separate validation data.

## DAVIS structural subset
Structural preprocessing may retain fewer targets because contact/alignment
assets are required. `prepare --allow-missing` opts into a subset and writes its
actual population to `filter_report.json`. Never claim full-benchmark coverage
for a filtered structural subset.

## Ligand base feature terminology
Historical code uses 44 atom-symbol bits + 11 degree + 11 total-H + 11 `GetImplicitValence()` + 1 aromaticity = 78. The manuscript's "implicit hydrogen count" description does not match the code.

## SMILES standardization
Historical graph code validates/parses SMILES but does not implement all salt/counter-ion/duplicate standardization steps described in the manuscript. `notebook` mode preserves that behavior; `canonical` and `fragment_parent` are explicit alternative experiments.

## FG overlap
Molecule-level metadata applies cleanup (Ester>Ether, CarboxylicAcid>Alcohol, Phenol>Alcohol). Historical per-atom ligand FG bits do not call that cleanup. `fg_overlap_policy=notebook` reproduces this behavior.

## Protein profile
The historical 21-D profile is a pseudocount-smoothed residue-frequency/probability profile, not a conventional log-odds PSSM.

## Protein FGs
Historical residue-FG features apply SMARTS to isolated free-amino-acid SMILES. Shared amino/carboxyl backbone groups can therefore be common/redundant. The ligand-only/protein-only ablation is intended to test this directly.

## Contact graph
Historical protein edges use contact probability >= 0.5. Top-k is implemented only as an explicit alternative.

## Exact historical stacked RGB image
Recovered from `onehot+PE.ipynb`:

- `aa_list = list("ACDEFGHIKLMNPQRSTVWYX")`
- `d_model = 32`
- `pe1d = sinusoidal_encoding(L, d_model)`
- row/column PE maps and row/column one-hot maps are concatenated
- `feat_map = combined.mean(axis=0)`
- R = independently min-max-normalized contact map
- G = independently min-max-normalized `feat_map`
- B = exactly zero
- saved at native LxL using `plt.imsave(..., origin="lower")`

This supersedes any earlier proposed non-historical channel scheme.

## Image sizes
AE: RGB -> Resize(512,512). PV preparation: RGB -> `Image.resize((256,256))` with omitted resample argument -> torchvision Resize(224,224) -> ImageNet normalization.

## ResNet
Historical full PV uses ImageNet-pretrained ResNet-101 and fine-tunes it by default.

The current repository default is `fine_tune_resnet=False`: backbone parameters
are frozen, while the projection and affinity layers remain trainable. BatchNorm
running statistics still follow the model's training/evaluation mode. The Python
constructor accepts `fine_tune_resnet=True` to enable backbone parameter updates.

## AE
Historical defaults: 512x512 RGB, 128-D latent, masked MSE where target > 0.05, Adam lr 0.001, batch size 8, 200 epochs. The later/final decoder reconstructs R/G and appends exactly zero B. AE latents are exported before DTA training and remain fixed.

## PV objective
Image gate: `z_image = z_resnet * z_ae`. InfoNCE aligns ligand/protein/image pairs symmetrically at tau=0.07. Full objective: MSE + 0.5*InfoNCE.

## Metrics
Repository evaluation uses one exact O(n log n) CI implementation for all models, plus MSE, Pearson and Spearman from saved predictions.

## Statistical tests
No values are pre-populated. The final CLI provides matched-seed mean/sample SD,
paired seed tests, target-MSE tests after averaging over seeds, bootstrap intervals,
and Holm adjustments. See STATISTICS.md for precise pairing and units.

## Baseline provenance
Literature-reported results must be labeled separately from repository reruns. Direct percentage improvements should use matched-protocol reruns only.
