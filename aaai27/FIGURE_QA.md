# Figure QA Record

Backend: Python with Matplotlib. The figures were regenerated from
`reports/v2_fc_paper_suite` and
`aaai27/figures/data/timeseries_overview.npz`.

## Evidence and uncertainty

| Figure | Evidence source | Uncertainty shown |
| --- | --- | --- |
| 1(a) | measured seed-7 per-timestep export, in-distribution task, `encoder_dropout` | none claimed; single window |
| 1(b) | protocol-level suite aggregates, nine operators / two splits / three seeds | cross-seed SD on the primary row |
| 2 | Both panels are explicitly illustrative; no numerical claim is read off them | none claimed |
| 3 | frozen detector results over nine faults and two task splits | cross-seed SD, `n=3`, in panel (b) |
| 4 | gated and ungated replay metrics | paired means over three seeds; no inferential claim |
| S1 | complete split-specific detectability matrix | mean over seeds 7, 13, 23 |
| S2 | ordered ablation suite | cross-seed SD, `n=3` |
| S3(a) | two maximum-severity stress settings | cross-seed SD, `n=3` |
| S3(b) | 15 leave-one-subject-out effects | percentile-bootstrap 95% CI, 10,000 resamples |

Figures S1--S3 are the supplement's Figures 1--3. The main paper carries four
figures; the teaser is Figure 1 and the method overview Figure 2.

The plotted values are observational outputs from the archived suite; no visual
element is manually adjusted to change the numerical conclusion.

## AAAI presentation checks

- Full-width figures use the 177.8 mm text width; the teaser and the stress
  figure use the 85.09 mm column width. The teaser is single-column by necessity:
  a `figure*` is a double-column float and LaTeX will not place one on the page
  carrying the title block, so a full-width teaser is always deferred to page 2.
- Every figure is placed at 1:1, so the size in the source is the size on paper.
  `bbox="tight"` crops the canvas to its content, so the exported width does not
  equal `figsize` and `\includegraphics` then rescales by an unpredictable factor:
  the three main figures were once cropped to 5.90/6.88/6.55 in, placed at
  `0.72\textwidth`, and printed one source size of 9.2 pt as 7.86/6.74/7.08 pt.
  `fit_canvas_to_print_width` now converges the canvas so the cropped width equals
  the typeset width, and `assert_prints_one_to_one` fails the build when the
  exported width deviates by more than 0.01 in.
- Figure text is generated at 8.5 pt against 10 pt body text, measuring 8.48–8.54 pt
  on paper across all seven figures; plotted lines are at least 0.5 pt.
- Legend and box placement is measured, never hard-coded: `figure_legend_below`
  positions figure-level legends below the lowest rendered content. Six geometry
  guards run after the canvas resize and fail the build on clipping, overlapping
  titles, legends leaving their panel, legends over text (axis-level and
  figure-level), and annotations reaching into a neighbouring panel. Two collision
  classes are outside their reach and are checked by rendering the page: an x-label
  running into the next panel's title, and in-panel annotation text sitting on the
  plotted data. Both were present in an earlier teaser draft and were fixed by
  raising `hspace` and giving the gate band its own strip below the torque range.
- Color is paired with marker shape, fill, line style, or direct labels.
- Submission PDFs are PDF 1.5 and convert text to outlines, avoiding Type 3 and
  Identity-H fonts. Editable text remains in the companion SVG files.
- PNG previews are exported at 600 dpi. TIFF is not generated because the
  submission uses vector PDF for these line-art and plot figures.
- The AUROC heatmap uses a diverging map centered at chance (0.5), and every
  cell also carries its numerical value.

## Explicit exceptions

Figure 2(a) uses a fixed random seed to draw illustrative noise, and its caption
labels those traces as illustrative. The measured replay window lives in Figure
1(a), whose caption names the exact split, operator, seed, and participant
status. This separation prevents the illustrative trace from being read as
experimental evidence.

Figure 1 is a teaser, so it selects one window out of the protocol. The caption
discloses that this window's aligned-command retention is 0.66, below the 96.1%
clean average, making the selected window conservative rather than favourable,
and panel (b)'s fourth row reports the torque-rate cost alongside the three
benefit rows.

The manuscript uses only vector PDF figures, so a TIFF export is unnecessary
for the AAAI submission. A 600 dpi PNG is retained solely as a visual QA
preview.
