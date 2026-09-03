# Reproducibility notes

PathScaleBench uses the audited native-FOV same-center protocol
`pathscalebench_native_fov_same_center_v1`.

For each tissue location, 224×224-pixel patches are evaluated at native 40×,
10×, and 2.5× resolutions while preserving the same tissue center. The physical
field of view changes with magnification.

Statistical units:
- paired case-level analyses use the same 71 cases across PFMs;
- model-level correlations use the PFM as the statistical unit (n=8);
- cross-validation repeats are not treated as independent biological observations;
- pair-matched criterion-validity permutations preserve model/scale-pair structure.

RudolfV2 and RudolfV2-S were requested during revision but were not included
because gated-access requests were not approved during the revision period.

Raw TCGA WSIs, model checkpoints, and large embedding files are not redistributed.
