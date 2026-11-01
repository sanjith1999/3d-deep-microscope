# CVPR manuscript

Compile `main.tex` for the paper or `supplement.tex` for the supplement using
pdfLaTeX and BibTeX. In Overleaf, select either file as the main document.
Locally, run `make`, `make main` or `make supplement`.

- `sections/`: manuscript sections and supplementary text
- `settings/`: formatting, title settings and the CVPR template
- `figures/`: figures used by the documents
- `tables/`: main tables and `supplement/` tables
- `references.bib`: shared bibliography
- `docs/writing.md`: editing conventions

Paper versions are on `paper/cvpr/init-draft`, `paper/cvpr/struct-amb` and
`paper/cvpr/dec-agn-depth`. The original manuscript and extended experimental
record are on `paper/manuscript`. Experimental code is on the `base-mdl` and
`experiment/*` branches. Each paper version is a self-contained LaTeX project.
