# Reference results

This directory contains the reference outputs used to reproduce the analyses
and figures reported in the revised PathScaleBench manuscript.

## Embedding extraction audit

`reference/embedding_extraction_audit_manifest.csv` contains the complete
71-slide × 8-model × 3-scale extraction audit (1,704 entries), including the
number of patch embeddings and observed feature dimensionality.

## Main analyses

- `01b_*`: cross-scale CKA (Figure 2)
- `02_*`, `05_*`, `06_*`: relative information retention and model/cancer analyses (Figure 3)
- `07_*`, `08_*`, `08b_*`, `10_*`, `11_*`: cross-scale retrieval similarity (Figure 4)
- `12_*`, `14_*`, `16_*`: cancer retrieval Recall@K and neighborhood preservation (Figure 5)
- `18_*`: multidimensional benchmark summary and ranks (Figure 6)
- `19_*`, `30b_*`: representation metrics versus retrieval criterion validity (Figure 7)
- `20_*`, `21d_*`, `23_*`, `31b_*`: magnification-shift robustness analyses (Figure 8)
- `22_*`: metric complementarity
- `24_*`: retrieval/NPS hyperparameter sensitivity
- `25_*`: CKA sampling sensitivity
- `32_*`: training-regime and family-level analyses

## Statistical note

For cross-scale retrieval similarity, the final model × scale-pair interaction
inference uses generalized estimating equations (GEE; `08b_*`). Exploratory
MixedLM fits that produced degenerate variance estimates were not used for
final inference and are not included in this release.

Model-level summary ranks are descriptive and are not interpreted as a single
overall performance score.
