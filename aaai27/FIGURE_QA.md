# Figure QA Record

Backend: Python with Matplotlib. The figures were regenerated from
`reports/v2_fc_paper_suite` and
`aaai27/figures/data/timeseries_overview.npz`.

## Evidence and uncertainty

| Figure | Evidence source | Uncertainty shown |
| --- | --- | --- |
| 1 | Panels (a)--(b) are explicitly illustrative; panel (c) is a measured seed-7 per-timestep export | none claimed |
| 2 | frozen detector results over nine faults and two task splits | cross-seed SD, `n=3`, in panel (b) |
| 3 | gated and ungated replay metrics | paired means over three seeds; no inferential claim |
| 4 | ordered ablation suite | cross-seed SD, `n=3` |
| 5(a) | two maximum-severity stress settings | cross-seed SD, `n=3` |
| 5(b) | 15 leave-one-subject-out effects | percentile-bootstrap 95% CI, 10,000 resamples |

The plotted values are observational outputs from the archived suite; no visual
element is manually adjusted to change the numerical conclusion.

## AAAI presentation checks

- Full-width figures use the 177.8 mm text width; the stress figure uses the
  85.09 mm column width.
- Figure text is generated at 9.2 pt or larger, and plotted lines are at least
  0.5 pt.
- Color is paired with marker shape, fill, line style, or direct labels.
- Submission PDFs are PDF 1.5 and convert text to outlines, avoiding Type 3 and
  Identity-H fonts. Editable text remains in the companion SVG files.
- PNG previews are exported at 600 dpi. TIFF is not generated because the
  submission uses vector PDF for these line-art and plot figures.
- The AUROC heatmap uses a diverging map centered at chance (0.5), and every
  cell also carries its numerical value.

## Explicit exceptions

Figure 1(a) uses a fixed random seed to draw illustrative noise. Its caption
labels those traces as illustrative, while panel 1(c) is identified as the
measured export. This distinction prevents the illustrative trace from being
read as experimental evidence.

The manuscript uses only vector PDF figures, so a TIFF export is unnecessary
for the AAAI submission. A 600 dpi PNG is retained solely as a visual QA
preview.
