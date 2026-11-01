# Manuscript and extended experiments

`manuscript/` contains the general research paper. `experiments/` contains the
extended experimental record. Each is a self-contained LaTeX project with
`main.tex`, `sections/`, `settings/` and `figures/`.

Run `make` in this folder to build both documents, or `make -C manuscript` and
`make -C experiments` separately. In Overleaf, select the corresponding
`main.tex`. Both use pdfLaTeX; the manuscript also uses BibTeX.

Edit text in `sections/`, shared formatting and title details in `settings/`,
and the manuscript bibliography in `manuscript/references.bib`.
`docs/writing.md` describes the editing conventions.

The later CVPR versions are on `paper/cvpr/init-draft`, `paper/cvpr/struct-amb`
and `paper/cvpr/dec-agn-depth`. Experimental code is on `base-mdl` and
`experiment/*`. Historical experiment identifiers in the record refer to the
research runs described there; they are not additional build dependencies.
