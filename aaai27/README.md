# AAAI-27 LaTeX Draft

This directory contains a paper scaffold derived from the repository implementation and the two baseline papers in `../`.

Files:

- `main.tex`: single-file anonymous AAAI-27 manuscript source, as required by the official author kit.
- `references.bib`: initial bibliography.
- `OUTLINE.md`: Chinese story, section budget, code mapping, experiment matrix, and evidence gaps.
- `figures/`: manuscript-ready vector PDF figures, editable SVG sources, and 300 dpi PNG previews.
- `../../scripts/draw_aaai_figures.py`: deterministic source for all manuscript figures.
- `aaai2027.sty`, `aaai2027.bst`: unmodified files copied from the official AAAI-27 Author Kit dated May 2026.

Build with:

```bash
make
```

Regenerate only the figures with:

```bash
make figures
```

The script renders five figures: `fig_method_overview`,
`fig_detection_taxonomy`, `fig_safety_utility` and `fig_ablation_summary`
at the AAAI full text width (7.0 in, `figure*`), plus `fig_stress_transfer`
at single-column width (3.35 in, `figure`).

Figures follow one validated visual system: a four-role categorical palette
(assistance blue, sensing violet, fault red, moment green), a blue ordinal
ramp for staged quantities, and a blue-gray-red diverging ramp centred on
chance for AUROC. Every categorical pair also carries a secondary encoding
(marker shape, fill, or a direct label) so the green/red pair stays readable
under color-vision deficiency. Labels are publication-scale, grid and axes
are recessive hairlines, and fonts are embedded as TrueType rather than
Type 3 glyphs.

Numeric figure panels reproduce the values reported in `main.tex`; update
both together when final experiment results change. Panels marked
illustrative in the captions (the method-overview traces and timelines) are
schematic — a data-backed per-timestep fault timeline needs the server-side
export described in `DATA_REQUIREMENTS.md` §8.

or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current environment may need a TeX Live installation. Before submission, replace every `[TBD]`, remove internal TODO comments, rebuild the aggregate experiment tables, and inspect the log for overfull boxes. Do not modify the official `.sty` or `.bst` files.
