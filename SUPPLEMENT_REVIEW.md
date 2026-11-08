# Supplement review findings

Review date: 2026-11-05

## Scope and status

This records the source-level review of the supplementary manuscript, its tables, and consistency with the main paper. The review did not reproduce the experiments, compile the documents, or inspect the rendered figures.

R05, and R06 were corrected on 2026-11-06; the remaining findings are open. Source line numbers refer to the reviewed version and may shift during editing.

PDF locations were checked on 2026-11-06 against the existing `supplement.pdf` (modified 2026-11-06 00:51) and `main.pdf` (modified 2026-11-05 23:38). PDF line numbers below are the printed review-line numbers in the page margins, not LaTeX source lines. Tables and captions have no printed review-line numbers; those locations use the page, table number, and relevant cell or caption instead. Locations may change after recompilation. PDF text was extracted to verify these locations; this was not a visual layout review.

Those locations identify the original reviewed passages. Resolution records include PDF locations at the time of each fix.

The supplement's strengths include substantial supporting experiments, explicit reporting of failed criteria, and limitations on connectivity claims. The main concerns are numerical consistency, reproducibility, and distinguishing results from different experimental configurations.

## Resolution checklist

- [ ] R01 — Reconcile mismatch-coverage failures and the attribution to P18.
- [ ] R02 — Reconcile certificate-gain and solver-rerun counts.
- [ ] R03 — Reconcile the learned-plate and DMD width predictions.
- [ ] R04 — Verify that the cited references sufficiently specify the optical objectives.
- [x] R05 — Label aperture configurations consistently.
- [x] R06 — Qualify the fewer-measurements comparison.
- [ ] R07 — Complete the certification formulations.
- [ ] R08 — Correct the main-paper table reference.
- [ ] R09 — Reconcile the seed-1 TV improvement.
- [ ] R10 — Distinguish the exposure label from conventional SNR.
- [ ] R11 — Document experiment-specific phantom splits.
- [ ] R12 — Improve the supplement's organization and paragraph structure.

Resolve findings individually. For numerical discrepancies, consult the experiment records rather than choosing a replacement value from the prose or tables alone. Record the evidence and verification under the relevant finding before marking it resolved.

## R01 — Mismatch-coverage attribution contradicts the table

**Status:** Open  
**Type:** Confirmed internal inconsistency

**PDF locations:**

- **Supplement, page 23, Table S33, caption:** “All failures with the allowance come from phantom P18.” The caption is unnumbered.
- **Supplement, page 23, Table S33, bottom-right cell:** row “multiplicative,” 25 dB; column “+ mismatch (99.9%)”; value **20/24**. Table cells are unnumbered.
- Related protocol: **page 23, lines 686–693**.

**Sources:**
- [Supplement source](sections/supplement.tex), lines 470–475, “Fine-grid model mismatch.”
- [Mismatch coverage table](tables/supplement/mismatch_coverage.tex), lines 2–6.

The caption says all failures after adding the mismatch allowance come from P18. However, the multiplicative, 25 dB, 99.9% allowance entry reports coverage of **20/24**, meaning four failures. With eight phantoms and three seeds, P18 can account for at most three failures in that entry.

**Why it matters:** The caption explains failures as a single exceptional geometry. The table does not support that explanation as written.

**Resolution needed:** Check per-phantom, per-seed coverage records. Correct the table, attribution, or experiment description as appropriate, and identify every failing phantom.

**Resolution record:** Pending.

## R02 — Numerical accounting does not reconcile

**Status:** Open  
**Type:** Confirmed accounting gaps

**PDF locations:**

- **Supplement, page 16, right column, lines 518–520:** “Outcomes” reports 24 versus 33 certificates and nine gains.
- **Supplement, page 16, right column, lines 522–523:** the phantom breakdown totals ten; the sign-test statement continues through **line 525**.
- **Supplement, page 21, left column, lines 659–662:** 110 reruns versus 59 ambiguities, five certificates, and one boundary case. The full “Numerical validation” paragraph is **lines 658–667**.

**Sources:**
- [Supplement source](sections/supplement.tex), line 355, “Outcomes.”
- [Supplement source](sections/supplement.tex), line 453, “Numerical validation.”

Two separate counts need reconciliation:

1. **Certificate gains:** The text reports nine gained query instances. Its breakdown—P01, P03, and P17 in all three draws, plus P10 in one—sums to **ten**. The reported aggregate change from 24 to 33 certificates is nine.
2. **Solver reruns:** The text says 110 timed-out problems were rerun, then accounts for 59 ambiguities, five certificates, and one boundary case: **65 outcomes**, leaving 45 unexplained.

**Why it matters:** The first discrepancy affects the traceability of the phantom-level statistical analysis. The second prevents readers from reconstructing the numerical-validation accounting.

**Resolution needed:** Reconcile both statements with the raw records. If “problem,” “solver job,” and “query instance” represent different counting units, define them and report their relationship.

**Resolution record:** Pending.

## R03 — Width predictions do not follow from the stated correlations

**Status:** Open  
**Type:** Quantitative clarification or correction required

**PDF locations:**

- **Supplement, page 11, left column, lines 208–212:** “predicted 0.80, measured 0.79.”
- **Supplement, page 11, left column, lines 214–217:** DMD prediction **2.52**, described as relative to the random plate.
- **Supplement, page 11, Table S14:** correlations and measured widths; table rows have no printed review-line numbers.
- **Main paper, page 5, left column, lines 306–311:** the two-voxel bound used for the comparison.

**Sources:**
- [Supplement source](sections/supplement.tex), lines 186–193.
- [Main-paper decoder section](sections/decoders.tex), line 35.
- [Phase quantisation table](tables/supplement/phase_quantisation.tex).
- [Reduced-field objective comparison](tables/supplement/objectives_reduced.tex).

With random-plate correlation **0.868**, the stated two-column rule gives:

$$
\text{predicted relative width}
\approx \sqrt{\frac{1-0.868^2}{1-\rho_e^2}}.
$$

| Encoder | Correlation | Calculated prediction | Reported prediction |
|---|---:|---:|---:|
| Learned plate, seed 0 | 0.761 | **0.7654** | **0.80** |
| DMD | 0.967 | **1.9490** | **2.52** |

**Checks of other variants:**

- **0.80:** The four-level plate (correlation 0.781) predicts **0.7951 ≈ 0.80**, but measures **0.82**, not 0.79. Table S14's measured 0.80 entries belong to the eight-level plate and its 0.2 rad-error variant; they are not predictions.
- **2.52:** Comparing the DMD against either the eight-level trained plate or the Fisher-information plate (both correlation 0.767) gives **2.5185 ≈ 2.52**. This uses a trained reference, whereas the text specifies the random plate.

**Resolution needed:** Verify the encoder, reference, and averaging procedure used in the original calculations. Predicted and measured ratios must use the same encoder and reference. These numerical matches suggest possible explanations, not confirmed corrections.

**Resolution record:** Pending.

## R04 — Verify the specification of the cited optical objectives

**Status:** Open  
**Type:** Reference and implementation verification; not a confirmed omission

**PDF location:** **Supplement, page 14, left column, lines 377–383**, in Section S10, “Plate objectives.” The smooth-maximum and Fisher-information descriptions are at **lines 379–383**.

**Source:** [Supplement source](sections/supplement.tex), lines 287–292, “Plate objectives.”

The coherence objective cites **Miniscope3D [7]**. The Fisher-information objective cites **[5]** and describes Poisson depth information with unknown brightness accounted for through a Schur complement, using a negative mean-log objective.

These citations may supply the required equations and settings; repeating their derivations is not necessarily needed. The cited formulations have not yet been checked against the implementation, so the original claim that both objectives were underspecified was too definite.

**Resolution needed:** Verify that references [7] and [5], together with the supplement's descriptions, sufficiently specify the implemented objectives. If followed unchanged, a precise equation or section reference may suffice. Clarify only adaptations or parameters not established by the references, including how the objectives are applied to detector-integrated excitation signatures.

**Resolution record:** Pending.

## R05 — Aperture configurations are not consistently attached to results

**Status:** Resolved (2026-11-06)

**Type:** Experimental-configuration ambiguity

**PDF locations:**

- **Supplement, page 1, lines 025–050:** aperture explanation; the change to four times the field is discussed at **lines 041–048**, continuing from the left column into the right column.
- **Supplement, page 9, Table S13:** row `C = 8`, “no plate,” depth-error cell **0.342 µm**, and the table caption.
- **Supplement, page 2, Table S2:** row `C = 8`, “no plate,” box-constrained TV depth-error cell **0.737 µm**, and the aperture footnote in the caption. Both tables and their captions lack printed review-line numbers.
- **Supplement, page 18, left column, lines 604–610:** reduced-operator geometry; the optical-grid size appears at **lines 607–608**.

**Sources:**
- [Supplement source](sections/supplement.tex), line 10 and lines 142–156.
- [TV table](tables/supplement/total_variation.tex), lines 3–6.
- [Final encoder table](tables/supplement/encoders.tex), lines 15–18.
- [Supplement source](sections/supplement.tex), line 391, reduced-operator geometry.

The opening explains the change from a two-times-field to a four-times-field aperture for the final no-plate comparison. Later tables retain results from the smaller aperture without clearly identifying that configuration in their captions.

For example, at compression factor 8, no-plate TV depth error is **0.342 µm** in the TV table and **0.737 µm** in the final encoder table. The opening explanation can reconcile these values, but the tables are not self-contained.

The reduced-operator description also needs to clarify whether its stated **96 × 96 optical grid at 0.08 µm** describes the specimen crop or the entire illuminated aperture.

**Resolution needed:** Label aperture multipliers in every affected comparison, tuning table, and operator diagnostic. Distinguish the simulation grid, illuminated aperture, and specimen crop in the geometry description.

**Resolution record:** Specified the two-times-field aperture in the operator, excitation, point-emitter, initial-tuning, expanded-tuning, and TV captions, and in the robustness section and figure. The expanded-tuning and TV captions now point to Table S2 for the final four-times-field no-plate comparison. Clarified that the reduced field's 96 × 96 samples describe the specimen region, with the overfilled illumination simulated beyond that region before cropping. Numerical table entries were unchanged.

**Verification:** `latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex` completed successfully, with no undefined references/citations or overfull/underfull box warnings. Checked the rebuilt PDF text: aperture protocol on **page 6, lines 125–128**; captions of **Tables S12 and S13 on page 9**; reduced-field geometry on **page 18, lines 614–622**. This verifies the manuscript clarification, not an independent rerun of the optical experiments.

## R06 — “Matches” overstates the fewer-measurements result

**Status:** Resolved (2026-11-06)

**Type:** Statistical wording inconsistent with the main paper

**PDF locations:**

- **Supplement, page 11, left column, lines 221–226:** “Full size”; “matches” appears at **line 223**, with the learned-PnP means and interval at **lines 224–225**.
- **Main paper, page 6, right column, lines 436–445:** “Fewer measurements”; the qualified learned-PnP comparison is at **lines 443–445**.

**Sources:**
- [Supplement source](sections/supplement.tex), line 204.
- [Main-paper results](sections/results.tex), line 31.

The supplement says the learned plate with 32 frames “matches” the random plate with 256 frames. For learned PnP, the reported interval spans approximately −17% to +15%, which does not establish equivalence.

The main paper already distinguishes comparable mean errors from evidence of equivalence.

**Resolution needed:** Align the supplement with the main paper's qualified wording. Separate the learned-PnP comparison from the demonstrated improvement under pretrained PnP and the lower mean error under TV.

**Resolution record:** Replaced “matches” with comparable mean depth errors under learned PnP and explicitly stated that the evidence does not establish equivalence. Reported TV's lower mean error and pretrained PnP's 35% reduction separately, retaining the existing results and interval.

**Verification:** The supplement compiled successfully. The corrected “Full size” paragraph appears on **page 11, lines 225–233** of the rebuilt PDF; the equivalence qualification is at **line 230**. The wording now agrees with the main paper's interpretation.

## R07 — Certification formulations omit necessary definitions

**Status:** Open  
**Type:** Mathematical specification gap

**PDF locations:**

- **Supplement, page 17, right column, lines 565–567:** enlarged consistency-set definition.
- **Supplement, page 17, right column, lines 570–575:** separator formulation; `t_lo` appears in Equation (S1) at **line 575**.
- **Supplement, page 17, right column, lines 581–585:** flow formulation; `t_hi` appears at **line 584**.
- **Supplement, page 17, right column, lines 586–587**, continuing on **page 18, left column, lines 588–592:** decision versions and witness margin `mu`.

**Source:** [Supplement source](sections/supplement.tex), lines 373–386, “Formulation details.”

The separator and flow formulations introduce `t_lo`, `t_hi`, and the witness margin `mu` without defining their values or construction. The admissible interval for `t` should be explicit because it determines the validity of the big-M constants.

The displayed definition of the enlarged consistency set also omits the intensity box, although surrounding prose describes bounded intensities.

**Resolution needed:** Define all bounds and margins, state the domain of `t`, and explicitly retain the intensity box in the enlarged set. Verify the written formulation against the implemented solver.

**Resolution record:** Pending.

## R08 — Main-paper table reference is stale

**Status:** Open

**Type:** Cross-reference error

**PDF locations:**

- **Supplement, page 14, right column, line 391:** “At full size (Tab. 3).”
- **Main paper, page 8, left column:** **Table 2** is “Plate objectives,” and **Table 3** is “Depth intervals and bounds.” These tables and captions lack printed review-line numbers.

**Sources:**
- [Supplement source](sections/supplement.tex), line 290.
- [Main-paper results](sections/results.tex), lines 21–67.

The plate-objectives paragraph says “At full size (Tab. 3).” In the current main-paper ordering, the objectives table is **Table 2**; Table 3 contains depth intervals and bounds.

**Resolution needed:** Correct the reference and check other manually numbered main-paper references against the current document.

**Resolution record:** Pending.

## R09 — Seed-1 TV improvement differs between tables

**Status:** Open  
**Type:** Numerical reporting discrepancy

**PDF locations:**

- **Supplement, page 12, left column, Table S16:** TV, seed `s1`, “all 60” depth-change cell: **−42%**.
- **Main paper, page 6, left column, Table 1:** TV row, “seed 1” depth-change cell: **−43%**.
- Both cells are in tables without printed review-line numbers.

**Sources:**
- [Extended-crops table](tables/supplement/additional_crops.tex), line 6.
- [Main decoder table](tables/decoders.tex), line 5.

For seed 1 with TV over all 60 crops, the supplementary table reports **−42%**, while the main table reports **−43%**.

**Resolution needed:** Check the underlying statistic and rounding convention, then make both tables consistent. If they use different statistics, label that difference explicitly.

**Resolution record:** Pending.

## R10 — Exposure-label and SNR terminology are mixed

**Status:** Open

**Type:** Units and terminology inconsistency

**PDF locations:**

- **Supplement, page 17, left column, lines 531–533:** “1.9 dB of SNR” and “about 2.4 times the photons.”
- **Supplement, page 18, left column, lines 622–634:** “Gaussian noise convention”; the standard-deviation formula is at **lines 626–627**, and the exposure-label qualification is at **lines 632–634**.

**Sources:**
- [Supplement source](sections/supplement.tex), line 355.
- [Supplement source](sections/supplement.tex), line 394, “Gaussian noise convention.”

The noise convention correctly defines `q` as a reference exposure label rather than conventional power-ratio SNR, with noise standard deviation proportional to `10^(-q/10)`. Elsewhere, a 1.9-unit improvement is called “1.9 dB of SNR” and equated to approximately 2.4 times the photons.

At fixed signal, a 1.9-unit increase in `q` corresponds to approximately **3.8 dB of conventional power SNR**, consistent with the stated 2.4-fold photon increase.

**Resolution needed:** Distinguish the exposure parameter from conventional SNR wherever it is discussed, including table headings and quantitative equivalences.

**Resolution record:** Pending.

## R11 — Experiment-specific phantom splits need a manifest

**Status:** Open  
**Type:** Documentation clarification; not a finding of data leakage

**PDF locations:**

- **Supplement, page 16, right column, lines 515–517:** the 24-phantom suite and three noise seeds.
- **Supplement, page 19, Table S28:** the full list of 24 phantoms; **page 21, Table S29:** the 13 reconstruction/connectivity cases. These tables lack printed review-line numbers.
- **Supplement, page 23, lines 688–693:** mismatch calibration; **Table S33's unnumbered caption** on the same page specifies eight held-out phantoms and three seeds.
- **Supplement, page 24, right column, lines 714–715**, continuing on **page 25, lines 716–717:** the ten named calibration phantoms and the quantities tuned using them.

**Sources:**
- [Supplement source](sections/supplement.tex), lines 353, 470–475, and 513.
- [Phantom definitions](tables/supplement/phantoms.tex).
- [Reconstruction/connectivity cases](tables/supplement/connectivity_cases.tex).

The ten named calibration IDs also appear in the overall 24-phantom suite. This alone does not establish leakage: the 13-case reconstruction table appears to exclude those calibration IDs, and not every experiment uses calibrated quantities.

However, the eight held-out mismatch phantoms are not identified in that section, and the relationship among the different subsets is not explicit enough.

**Resolution needed:** List calibration and evaluation IDs separately for each structural experiment, including the eight mismatch phantoms. Distinguish the full diagnostic suite from held-out evaluations of calibrated quantities.

**Resolution record:** Pending.

## R12 — Organization obscures the main contribution

**Status:** Open  
**Type:** Editorial recommendation

**PDF locations:** This is a document-wide finding rather than a single-line error. Representative anchors are:

- **Supplement, page 1, lines 001–006:** the opening roadmap.
- **Page 11, line 187:** Section S5, “Learned plate.”
- **Page 12, lines 239 and 279:** Section S6, “Extended test set,” and Section S8, “Learned prior and hybrid.”
- **Page 13, right column, lines 331–336:** Section S9 introduces the unused diffusion decoder.
- **Page 14, left column, lines 376–383:** Section S10 returns to plate objectives.
- **Page 24, left column, lines 694–697:** Section S22, “Preliminary single-stack study.”

**Source:** [Supplement source](sections/supplement.tex), overall organization.

The current ordering interrupts the evidence for decoder-free optical learning with historical experiments. Several long paragraphs combine protocol, results, statistical interpretation, and limitations.

**Suggested revision:** Group learned-plate training, decoder configuration, objective comparisons, and extended-test results near the beginning. Move the unused diffusion decoder and preliminary single-stack study toward the end. Separate protocol from outcomes in dense sections, while retaining negative results and qualifications.

**Resolution record:** Pending.
