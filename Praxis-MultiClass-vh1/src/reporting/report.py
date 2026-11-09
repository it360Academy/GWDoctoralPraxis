"""Assembles outputs/reports/H1_validation_report.html from the tables
and figures already written by the model/report stages."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

H1_TEXT = (
    "Within a leakage-controlled, cross-year-replicated supervised multiclass "
    "framework for diagnostic-quality auditing, a hospitalized patient's "
    "secondary-diagnosis profile (DX2-DX25) discriminates the principal "
    "diagnosis (DX1) among fifteen vascular, infectious, and oncological "
    "phenotypes in the U.S. National Inpatient Sample, 2021-2023, under "
    "severe class imbalance."
)


def _df_html(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x)
                       if isinstance(x, float) else x, border=0, classes="tbl")


def build_report(cfg, verdict: dict, phenotype_table: pd.DataFrame,
                  model_perf: pd.DataFrame, leakage_audit: pd.DataFrame,
                  negative_control: dict, best_algorithm: str) -> Path:
    out = cfg.outputs("reports") / "H1_validation_report.html"
    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #222; }
      h1, h2 { color: #14304d; }
      .tbl { border-collapse: collapse; margin: 12px 0; }
      .tbl th, .tbl td { border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }
      .tbl th { background: #f0f4f8; }
      .verdict { font-size: 20px; font-weight: bold; padding: 10px; border-radius: 6px; }
      .supported { background: #e3f6e3; color: #1c6b1c; }
      .not-supported { background: #fbe7e7; color: #9c2b2b; }
      .caveat { background: #fff8e1; border-left: 4px solid #e0a800; padding: 10px 16px; margin: 16px 0; }
      img { max-width: 900px; border: 1px solid #ddd; }
    </style>
    """
    verdict_class = "supported" if verdict["H1_SUPPORTED"] else "not-supported"
    verdict_text = "H1: SUPPORTED" if verdict["H1_SUPPORTED"] else "H1: NOT SUPPORTED"

    html = [f"<html><head><meta charset='utf-8'>{css}</head><body>"]
    html.append(f"<h1>H1 Validation Report</h1>")
    html.append(f"<p><i>Generated {datetime.now().isoformat(timespec='seconds')}</i></p>")
    html.append(f"<p><b>H1:</b> {H1_TEXT}</p>")
    html.append(f"<div class='verdict {verdict_class}'>{verdict_text}</div>")
    html.append("<div class='caveat'><b>Read before trusting this report:</b> "
                 "disease definitions are DRAFT/unverified, all h1_criteria floors "
                 "are unverified placeholders, and this may be a smoke run on "
                 "synthetic data - see README.</div>")

    html.append("<h2>Section 0: best-performing algorithm (selected on validation year)</h2>")
    html.append(f"<p>{best_algorithm}</p>")

    html.append("<h2>Section 1: aggregate test-year metrics</h2>")
    agg_rows = [
        {"metric": "Macro AUROC (OVR)", "value": verdict["macro_auroc_ovr"], "95% CI": verdict["macro_auroc_ovr_ci"]},
        {"metric": "Macro MCC", "value": verdict["macro_mcc"], "95% CI": verdict["macro_mcc_ci"]},
        {"metric": "Balanced accuracy", "value": verdict["balanced_accuracy"], "95% CI": verdict["balanced_accuracy_ci"]},
    ]
    html.append(_df_html(pd.DataFrame(agg_rows)))

    html.append("<h2>Section 2: negative control (permutation test)</h2>")
    html.append(_df_html(pd.DataFrame([negative_control])))

    html.append("<h2>Section 3: per-phenotype criteria</h2>")
    html.append(_df_html(phenotype_table))

    html.append("<h2>Section 4: algorithm x year performance</h2>")
    html.append(_df_html(model_perf))

    html.append("<h2>Section 5: leakage audit (masked codes)</h2>")
    html.append(f"<p>{len(leakage_audit):,} secondary-position codes masked from the "
                f"feature vocabulary for matching a phenotype definition.</p>")
    html.append(_df_html(leakage_audit.head(50)))

    html.append("<h2>Figures</h2>")
    html.append("<p><img src='../plots/confusion_matrix_test.png'></p>")
    html.append("<p><img src='../plots/per_class_auroc_test.png'></p>")
    html.append("<p><img src='../plots/algorithm_comparison.png'></p>")

    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    return out
