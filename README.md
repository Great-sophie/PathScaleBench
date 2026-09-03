# PathScaleBench

PathScaleBench is a representation-level benchmark for characterizing how frozen
pathology foundation model (PFM) representations change across histological
magnifications and how those changes relate to downstream magnification-shift robustness.

## Revised benchmark
- 71 TCGA WSIs
- 8 cancer types
- 8 PFMs: Phikon, UNI, Virchow, Virchow2, UNI2, Midnight, GigaPath, GigaPath-Flash
- Native-FOV same-center evaluation at 40×, 10×, and 2.5×

## Benchmark dimensions
1. Global alignment (linear CKA)
2. Relative cross-scale retention
3. Bidirectional cross-scale retrieval similarity
4. Biological identity preservation (Recall@K)
5. Neighborhood Preservation Score (NPS)

Downstream validation uses balanced accuracy, magnification generalization gap
(MGG), and probabilistic shift robustness.

## Important terminology
This is a **native-FOV same-center** protocol, not a matched-FOV protocol.

## Repository structure
- `scripts/analysis/`: benchmark and statistical analyses
- `scripts/figures/`: figure reproduction
- `manifests/`: audited cohort and sampling manifests
- `results/reference/`: small frozen reference tables
- `configs/benchmark.yaml`: frozen benchmark configuration
- `docs/REPRODUCIBILITY.md`: reproducibility details

## Data
Raw TCGA WSIs and third-party PFM checkpoints are not redistributed. Obtain them
from their original providers and reconstruct the benchmark using the released manifests.

## Model-access note
RudolfV2 and RudolfV2-S were requested during revision but could not be included
because gated-access requests were not approved during the revision period.

## Citation
Citation information will be added upon publication.

## Native-FOV sampling manifest

`manifests/sampling/native_fov_same_center_triplets.csv.gz` contains the
complete patch-level sampling records for the 71-slide native-FOV same-center
benchmark (1,209,736 records). The corresponding 71-slide summary is provided
in `manifests/sampling/native_fov_sampling_summary.csv`.

The compressed manifest can be read directly with pandas:

```python
import pandas as pd
df = pd.read_csv(
    "manifests/sampling/native_fov_same_center_triplets.csv.gz"
)



```
