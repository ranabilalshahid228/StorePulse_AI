# StorePulse AI — presentation (Beamer, 16:9)

- `slides.tex` → `slides.pdf`: title + 17 content slides, then 5 backup slides and 2 reference
  slides; about 20–25 minutes at 1–1.5 minutes per content slide.
- `slides-notes.tex` → `slides-notes.pdf`: the same slides with the speaker notes on the right half
  of each page (presenter copy).
- Theme: `metropolis` with a restrained navy / blue palette matching the report's charts.
- Title slide: Mathematics of Business and Industry, supervisor Prof. Dr. Stefanie Vogl,
  OTH Regensburg logo (`figures/oth-regensburg-logo.png`).

## Build

```powershell
tectonic slides.tex
tectonic slides-notes.tex
# or: latexmk -pdf slides.tex   /   latexmk -xelatex slides.tex
```

Packages: beamer (metropolis theme), booktabs, graphicx, tikz (arrows.meta, positioning,
shapes.geometric, calc), xurl, pgfpages (notes version). Bibliography: `references.bib`
(a copy of the report's) with `IEEEtran.bst`.

## Notes

- To change the presentation date, edit the `\date` line in `slides.tex`.
- In `slides-notes.pdf` the table on “Result 1” can appear washed out in some PDF viewers (an
  xdvipdfmx/pgfpages colour-stack effect when slides are shrunk onto the notes page). `slides.pdf`,
  the version to project, renders correctly.

## Assets

All charts and screenshots are copies from `../report/figures/` (charts generated from
`../report/data/` by `../report/scripts/make_figures.py`; screenshots from the running app with
demo data).
