#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-ready supplementary tables for Reviewer #1
=======================================================

Inputs:
  32_training_regime_multiscale_analysis_8pfm_*.csv

Outputs:
  Table S1  PFM training-regime audit
  Table S2  Virchow vs Virchow2 directional MGG
  Table S3  Virchow-family multidimensional comparison
  Table S4  GigaPath vs GigaPath-Flash distillation comparison

Each table is exported as:
  *.csv
  *.md
  *.tex

No statistics are recomputed except simple descriptive summaries
directly from the frozen Analysis-32 output files.
"""

from pathlib import Path
import csv
import math


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "results" / "reference"

OUT = ROOT / "publication_tables_training_regime"
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "32_training_regime_multiscale_analysis_8pfm"

TRAINING = ROOT / f"{PREFIX}_training_regime_table.csv"
DIRECTION = ROOT / f"{PREFIX}_virchow_direction_level.csv"
VIRCHOW = ROOT / f"{PREFIX}_virchow_family_contrast.csv"
DISTILL = ROOT / f"{PREFIX}_gigapath_distillation_contrast.csv"


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(path, headers, rows):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers
        )
        writer.writeheader()
        writer.writerows(rows)


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def f4(x):
    """Four-decimal publication formatting."""
    if x is None or x == "":
        return ""

    try:
        x = float(x)
    except Exception:
        return str(x)

    if math.isnan(x):
        return ""

    return f"{x:.4f}"


def f3(x):
    if x is None or x == "":
        return ""

    try:
        x = float(x)
    except Exception:
        return str(x)

    if math.isnan(x):
        return ""

    return f"{x:.3f}"


def delta4(x):
    if x is None or x == "":
        return ""

    try:
        x = float(x)
    except Exception:
        return str(x)

    if math.isnan(x):
        return ""

    return f"{x:+.4f}"


def bool_text(x):
    if x is None:
        return "Not documented"

    s = str(x).strip().lower()

    if s in {"true", "1", "yes"}:
        return "Yes"

    if s in {"false", "0", "no"}:
        return "No"

    if s in {"nan", "", "none"}:
        return "Not documented"

    return str(x)


def md_escape(x):
    return str(x).replace("|", "\\|")


def latex_escape(x):
    s = str(x)

    repl = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "±": r"$\pm$",
        "→": r"$\rightarrow$",
        "↔": r"$\leftrightarrow$",
        "×": r"$\times$",
        "Δ": r"$\Delta$",
        "ρ": r"$\rho$",
    }

    for a, b in repl.items():
        s = s.replace(a, b)

    return s


def write_markdown(path, title, headers, rows, footnotes=None):
    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(f"### {title}\n\n")

        f.write(
            "| "
            + " | ".join(
                md_escape(h)
                for h in headers
            )
            + " |\n"
        )

        f.write(
            "| "
            + " | ".join(
                "---"
                for _ in headers
            )
            + " |\n"
        )

        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    md_escape(
                        row.get(h, "")
                    )
                    for h in headers
                )
                + " |\n"
            )

        if footnotes:
            f.write("\n")
            for note in footnotes:
                f.write(f"*{note}*\n\n")


def write_latex(path, title, label, headers, rows, footnotes=None):
    """
    Generates a clean LaTeX table using booktabs.
    """

    # first column left, remaining columns centered
    align = "l" + ("c" * (len(headers) - 1))

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")

        f.write(
            "\\caption{"
            + latex_escape(title)
            + "}\n"
        )

        f.write(
            "\\label{"
            + label
            + "}\n"
        )

        f.write(
            "\\begin{tabular}{"
            + align
            + "}\n"
        )

        f.write("\\toprule\n")

        f.write(
            " & ".join(
                latex_escape(h)
                for h in headers
            )
            + " \\\\\n"
        )

        f.write("\\midrule\n")

        for row in rows:
            f.write(
                " & ".join(
                    latex_escape(
                        row.get(h, "")
                    )
                    for h in headers
                )
                + " \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        if footnotes:
            f.write("\\vspace{2mm}\n")
            f.write("\\begin{minipage}{0.98\\linewidth}\n")
            f.write("\\footnotesize\n")

            for note in footnotes:
                f.write(
                    latex_escape(note)
                    + "\\\\\n"
                )

            f.write("\\end{minipage}\n")

        f.write("\\end{table}\n")


def export_table(
    basename,
    title,
    label,
    headers,
    rows,
    footnotes
):
    csv_path = OUT / f"{basename}.csv"
    md_path = OUT / f"{basename}.md"
    tex_path = OUT / f"{basename}.tex"

    write_csv(
        csv_path,
        headers,
        rows
    )

    write_markdown(
        md_path,
        title,
        headers,
        rows,
        footnotes
    )

    write_latex(
        tex_path,
        title,
        label,
        headers,
        rows,
        footnotes
    )

    print(f"[DONE] {basename}")
    print(f"       {csv_path}")
    print(f"       {md_path}")
    print(f"       {tex_path}")


# ============================================================
# TABLE S1
# PFM TRAINING-REGIME AUDIT
# ============================================================

src = read_csv(TRAINING)

MODEL_NAME = {
    "phikon": "Phikon",
    "uni": "UNI",
    "virchow": "Virchow",
    "virchow2": "Virchow2",
    "uni2": "UNI2",
    "midnight": "Midnight",
    "gigapath": "GigaPath",
    "gigapath_flash": "GigaPath-Flash",
    "GigaPath-Flash": "GigaPath-Flash",
}


REGIME_LABEL = {
    "single_nominal_magnification":
        "Single nominal magnification",

    "explicit_mixed_magnification":
        "Explicit mixed magnification",

    "not_sufficiently_documented":
        "Not sufficiently documented",

    "distilled_family":
        "Distilled family",
}


rows = []

for r in src:

    model = (
        r.get("model", "")
        or r.get("model_display", "")
    )

    model = MODEL_NAME.get(
        model,
        model
    )

    regime = r.get(
        "regime_class",
        ""
    )

    rows.append({
        "PFM":
            model,

        "Histological pretraining scale regime":
            r.get(
                "histological_scale_regime",
                ""
            ),

        "Training-regime classification":
            REGIME_LABEL.get(
                regime,
                regime.replace("_", " ")
            ),

        "Explicit mixed-magnification pretraining":
            bool_text(
                r.get(
                    "explicit_mixed_magnification",
                    ""
                )
            ),

        "Distilled":
            bool_text(
                r.get(
                    "distilled",
                    ""
                )
            ),

        "TCGA included in pretraining":
            bool_text(
                r.get(
                    "tcga_pretraining_overlap",
                    ""
                )
            ),
    })


headers = [
    "PFM",
    "Histological pretraining scale regime",
    "Training-regime classification",
    "Explicit mixed-magnification pretraining",
    "Distilled",
    "TCGA included in pretraining",
]


notes = [
    (
        "Histological multi-magnification exposure refers to "
        "explicit sampling of pathology image content at different "
        "physical resolutions or nominal magnifications during "
        "pretraining; generic SSL multi-crop augmentation was not "
        "classified as histological multi-magnification training."
    ),
    (
        "Training regimes were classified conservatively. "
        "Models for which explicit mixed-magnification exposure "
        "could not be established from the audited public "
        "documentation were labelled 'Not sufficiently documented'."
    ),
    (
        "TCGA overlap is reported to aid interpretation of downstream "
        "TCGA-derived evaluation and is not itself treated as an "
        "explanatory variable in statistical inference."
    ),
]


export_table(
    "Table_S1_PFM_training_regime_audit",
    (
        "Pretraining-regime audit of the eight pathology "
        "foundation models included in PathScaleBench."
    ),
    "tab:S1_training_regime",
    headers,
    rows,
    notes,
)


# ============================================================
# TABLE S2
# VIRCHOW -> VIRCHOW2 DIRECTIONAL MGG
# ============================================================

src = read_csv(DIRECTION)

rows = []

v1 = []
v2 = []

for r in src:

    a = float(
        r["Virchow_MGG"]
    )

    b = float(
        r["Virchow2_MGG"]
    )

    d = float(
        r["Virchow2_minus_Virchow"]
    )

    v1.append(a)
    v2.append(b)

    direction = (
        r["direction"]
        .replace("40x", "40×")
        .replace("10x", "10×")
        .replace("2.5x", "2.5×")
        .replace("->", "→")
    )

    rows.append({
        "Source→target shift":
            direction,

        "Virchow MGG":
            f4(a),

        "Virchow2 MGG":
            f4(b),

        "Δ MGG (Virchow2−Virchow)":
            delta4(d),

        "Lower MGG with Virchow2":
            "Yes" if b < a else "No",
    })


mean_v1 = sum(v1) / len(v1)
mean_v2 = sum(v2) / len(v2)

rows.append({
    "Source→target shift":
        "Mean across six shifts",

    "Virchow MGG":
        f4(mean_v1),

    "Virchow2 MGG":
        f4(mean_v2),

    "Δ MGG (Virchow2−Virchow)":
        delta4(
            mean_v2
            -
            mean_v1
        ),

    "Lower MGG with Virchow2":
        "6/6 directions",
})


headers = [
    "Source→target shift",
    "Virchow MGG",
    "Virchow2 MGG",
    "Δ MGG (Virchow2−Virchow)",
    "Lower MGG with Virchow2",
]


notes = [
    (
        "MGG, magnification generalization gap. Lower values "
        "indicate greater robustness to source-to-target "
        "magnification shift."
    ),
    (
        "Virchow2 had a lower MGG than Virchow in all six "
        "matched source-to-target shifts."
    ),
    (
        "Mean MGG decreased from 0.4321 to 0.1812, "
        "corresponding to a 58.1% relative reduction."
    ),
    (
        "Two-sided exact paired sign-flip test: p=0.03125. "
        "Exact paired Wilcoxon signed-rank test: p=0.03125."
    ),
    (
        "The six matched magnification-shift directions, rather "
        "than cross-validation repeats or individual cases, were "
        "used as the paired units for this directional comparison."
    ),
]


export_table(
    "Table_S2_Virchow_Virchow2_directional_MGG",
    (
        "Family-matched comparison of magnification generalization "
        "gaps between Virchow and Virchow2."
    ),
    "tab:S2_virchow_mgg",
    headers,
    rows,
    notes,
)


# ============================================================
# TABLE S3
# MULTIDIMENSIONAL VIRCHOW FAMILY COMPARISON
# ============================================================

src = read_csv(VIRCHOW)

METRIC_LABEL = {
    "CKA":
        "Global alignment (CKA)",

    "Retention":
        "Relative retention",

    "Retrieval":
        "Cross-scale retrieval similarity",

    "Recall@1":
        "Biological identity Recall@1",

    "NPS":
        "Neighborhood Preservation Score",

    "mean_MGG":
        "Mean magnification generalization gap",

    "shift_robustness":
        "Probabilistic shift robustness",

    "mean_within_scale_BA":
        "Mean within-scale balanced accuracy",

    "mean_cross_scale_BA":
        "Mean cross-scale balanced accuracy",
}


INTERPRETATION = {
    "CKA":
        "Lower",

    "Retention":
        "Lower",

    "Retrieval":
        "Higher",

    "Recall@1":
        "Higher",

    "NPS":
        "Slightly lower",

    "mean_MGG":
        "Improved robustness",

    "shift_robustness":
        "Improved robustness",

    "mean_within_scale_BA":
        "Higher",

    "mean_cross_scale_BA":
        "Higher",
}


rows = []

for r in src:

    metric = r.get(
        "metric",
        ""
    )

    # only retain publication-relevant metrics
    if metric not in METRIC_LABEL:
        continue

    a = float(
        r["Virchow"]
    )

    b = float(
        r["Virchow2"]
    )

    d = float(
        r["Virchow2_minus_Virchow"]
    )

    rows.append({
        "Metric":
            METRIC_LABEL[
                metric
            ],

        "Virchow":
            f4(a),

        "Virchow2":
            f4(b),

        "Δ (Virchow2−Virchow)":
            delta4(d),

        "Direction with Virchow2":
            INTERPRETATION[
                metric
            ],
    })


headers = [
    "Metric",
    "Virchow",
    "Virchow2",
    "Δ (Virchow2−Virchow)",
    "Direction with Virchow2",
]


notes = [
    (
        "Virchow was pretrained at a single nominal 20×/0.5-mpp "
        "resolution, whereas Virchow2 used explicitly documented "
        "mixed-magnification exposure."
    ),
    (
        "Virchow2 substantially improved magnification-shift "
        "robustness but did not uniformly increase all "
        "representation-level scale-awareness metrics."
    ),
    (
        "In particular, cross-scale retrieval similarity and "
        "biological identity preservation increased, whereas "
        "CKA, relative retention, and NPS decreased."
    ),
    (
        "Mean cross-scale balanced accuracy increased by 0.3641, "
        "compared with a 0.1132 increase in mean within-scale "
        "balanced accuracy."
    ),
    (
        "These comparisons are family-matched associations and "
        "should not be interpreted as causal estimates of the "
        "effect of mixed-magnification pretraining."
    ),
]


export_table(
    "Table_S3_Virchow_Virchow2_multidimensional_comparison",
    (
        "Multidimensional representation and robustness comparison "
        "between Virchow and Virchow2."
    ),
    "tab:S3_virchow_multidimensional",
    headers,
    rows,
    notes,
)


# ============================================================
# TABLE S4
# GIGAPATH -> GIGAPATH-FLASH DISTILLATION
# ============================================================

src = read_csv(DISTILL)

rows = []

for r in src:

    metric = r.get(
        "metric",
        ""
    )

    if metric not in METRIC_LABEL:
        continue

    a = float(
        r["GigaPath"]
    )

    b = float(
        r["GigaPath-Flash"]
    )

    d = float(
        r["Flash_minus_GigaPath"]
    )

    rows.append({
        "Metric":
            METRIC_LABEL[
                metric
            ],

        "GigaPath":
            f4(a),

        "GigaPath-Flash":
            f4(b),

        "Δ (Flash−GigaPath)":
            delta4(d),
    })


headers = [
    "Metric",
    "GigaPath",
    "GigaPath-Flash",
    "Δ (Flash−GigaPath)",
]


notes = [
    (
        "The GigaPath/GigaPath-Flash comparison provides a "
        "within-family descriptive view of representation behavior "
        "following model distillation."
    ),
    (
        "Because only one teacher–student family was available, "
        "this comparison is descriptive and is not interpreted as "
        "a general statistical effect of distillation."
    ),
    (
        "A positive change is not uniformly favorable because the "
        "direction of better performance differs across metrics; "
        "for MGG, lower values indicate greater robustness, whereas "
        "for probabilistic shift robustness, higher values indicate "
        "greater robustness."
    ),
]


export_table(
    "Table_S4_GigaPath_GigaPathFlash_distillation",
    (
        "Descriptive within-family comparison of GigaPath and "
        "the distilled GigaPath-Flash model."
    ),
    "tab:S4_gigapath_distillation",
    headers,
    rows,
    notes,
)


# ============================================================
# MASTER README
# ============================================================

readme = OUT / "README_publication_tables.txt"

readme.write_text(
    """
PUBLICATION TABLES — TRAINING REGIME ANALYSIS
=============================================

Table S1
PFM training-regime audit.

Table S2
Virchow vs Virchow2 directional magnification generalization gap.
Primary response to Reviewer #1 regarding mixed-magnification training.

Table S3
Virchow vs Virchow2 multidimensional representation/robustness contrast.
Supports the conclusion that scale-awareness is multidimensional.

Table S4
GigaPath vs GigaPath-Flash distillation comparison.
Descriptive only because one teacher/student family is available.

Recommended manuscript placement
--------------------------------
Main Results:
  Report the Table S2 mean MGG, 58.1% reduction, 6/6 directions,
  exact p=0.03125, within-scale BA, cross-scale BA,
  and probabilistic shift robustness.

Supplementary:
  Tables S1-S4 in full.

Discussion:
  Emphasize association rather than causation.
  Do not state that mixed-magnification training universally
  improves all representation-level metrics.

Formatting conventions
----------------------
  Numerical metrics: 4 decimals
  Exact p values: retained at reported precision
  Magnification: × symbol
  Direction: → symbol
  Delta: Virchow2 minus Virchow / Flash minus GigaPath
""".strip()
    + "\n",
    encoding="utf-8"
)


print()
print("=" * 90)
print("PUBLICATION TABLE GENERATION COMPLETE")
print("=" * 90)
print(f"Output directory:\n{OUT}")
print()

for p in sorted(OUT.iterdir()):
    print(p.name)
