# StorePulse AI — presentation (Beamer, 16:9)

- `slides.tex` → `slides.pdf`: title + 14 content slides, then 4 backup slides and 2 reference slides, planned for
  **20 minutes** (placeholder — adjust to the required duration; about 1–1.5 min per main slide).
- `slides-notes.tex` → `slides-notes.pdf`: the same slides with the speaker notes on the right half
  of each page (presenter copy).
- Theme: `metropolis` with a restrained navy / blue palette matching the report's charts.

## Build

```powershell
tectonic slides.tex
tectonic slides-notes.tex
# or: latexmk -pdf slides.tex   /   latexmk -xelatex slides.tex
```

Packages: beamer (metropolis theme), booktabs, graphicx, tikz (arrows.meta, positioning,
shapes.geometric, calc), xurl, pgfpages (notes version). Bibliography: `references.bib`
(a copy of the report's) with `IEEEtran.bst`.

## Before presenting

- **Logo:** the title slide shows a red, clearly labelled placeholder. Replace it with the official
  OTH Regensburg logo file supplied by the university (follow its brand guidelines for clear space
  and minimum size): put the file in `figures/` and change the `\titlegraphic{...}` block in
  `slides.tex` to `\titlegraphic{\hfill\includegraphics[height=1cm]{figures/oth-logo.pdf}}`.
- **Placeholders (red):** programme, supervisor, faculty and campus, date, duration — in the
  `\institute` and `\date` lines of `slides.tex`.
- **Notes PDF caveat:** in `slides-notes.pdf` the table on “Result 1” can appear washed out in some
  PDF viewers (an xdvipdfmx/pgfpages colour-stack effect when slides are shrunk onto the notes page).
  `slides.pdf`, the version to project, renders correctly.

## Assets

All figures are copies from `../report/figures/` (charts generated from `../report/data/` by
`../report/scripts/make_figures.py`; screenshots from the running app with demo data). Regenerate
them in the report folder and copy them here.
