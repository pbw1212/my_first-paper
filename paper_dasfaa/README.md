# Recovering Both Diffusion Sources from a Late Cascade Suffix

The anonymous submission is [main.pdf](main.pdf). Build [main.tex](main.tex), which includes the experiments in [evaluation.tex](evaluation.tex). This directory includes the Springer LNCS class and bibliography style, [references.bib](references.bib), six vector figures, and six LaTeX tables.

## Rebuild

From this directory, run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The included figures and tables suffice to build the paper without the experimental archive. The LaTeX build uses accsupp to make a few displayed math symbols explicit when copying text from the PDF.

From the repository root, the figure generator uses the tracked summaries and window totals. The table generator and table audit also read the locally archived per-event files from their original paths under runs/. See [data and reproduction](../docs/reproduction.md) for the archive layout. The final command takes an optional directory containing the LaTeX log, bibliography, auxiliary file and PDF when building out of tree.

```text
python -B scripts/build_dasfaa_figures.py
python -B scripts/build_dasfaa_tables.py
python -B scripts/verify_dasfaa_tables.py
python -B scripts/verify_dasfaa_pdf.py
```

## Evidence and interpretation

| Exhibit | Primary record and reconstruction |
| --- | --- |
| Table 1 | The 64 columns are constructed by the feature pipeline ending in PublicWindow.design in distilled_source_pairs.py. The tier masks come from recent_member_evidence.py. [Feature index](evidence/feature_columns.json) gives exact implementation expressions, archived names where defined, and the tier in which each column first appears. |
| Table 2 | [Inventory](evidence/inventory.json) records distinct nodes or posting addresses, cleaned contacts or messages, and periods. The archive does not supply all of those counts in one summary. With the original input archive available to the repository readers, dasfaa_archive_counts.py recounts them. Evaluation windows and eligible counts come from WP1, WP2 and WP5. The upper panel distinguishes local fit/tune counts, transferred weights and test events. |
| Table 3 | Both@12 and Any@12 use the same WP1 events. The lower panel also gives MLP per-source recall and mean legal pool size. WP1 supplies the MLP Any@12 intervals and exact-pair ranks. The matching snapshot evaluation is recorded in [Top-1 audit](evidence/snapshot_top1.json). Tree and FGSSI ranks, static lists, MLP ranks and single-fit GCNSI predictions are checked by event ID. [Rates and paired GNU libc contrast](evidence/top12.json). |
| Table 4 | WP3 stress results provide the rule-specific and common-draw populations, paired means and both interval endpoints. |
| Table 5 | The heldout_r_lists/early_only/role_evidence/closure summary records 1,793 D0 threads. Four method columns count joint hits. Two additional columns divide full-model hits by eligible and reachable threads. The last column reports paired full-minus-posts differences with 95% intervals among eligible threads. The one-visible-member threshold applies to Role-Full and the other role variants. Positive contrasts with a strictly positive interval lower bound are marked in bold. |
| Table 6 | The WP5 real/common results hold the discussion IDs fixed across all three depths. Sim-MLP denotes the frozen simulation-trained temporal MLP. The three-visible-member scoring threshold applies at every depth. Unscored reports abstentions in depth order. |
| Figs. 1–3 | Fig. 1 is a schematic of the observation protocol. WP1 and WP2 supply the numerical budget and selectivity panels. |
| Fig. 4 | trained_source_mixtures supplies the empirical means. Its summary contains paired contrast intervals. Marginal method intervals are reconstructed from archived Enron hits using the original seed and three-window block procedure. [Window totals](evidence/prior_window_hits.json) permit regeneration without the event archive. The reconstruction also reproduces the archived paired intervals. source_prior_shift supplies the separate complete-trace ceiling components. |
| Figs. 5–6 | WP4 supplies unknown-count strata and 95% intervals for the coverage and recall curves. WP5 supplies the paired sender-prior and first-recipient contrasts on common draws. |

[Table audit](evidence/table_audit.json) records the row and cell checks. [Compiled PDF checks](evidence/build_check.json) record the manuscript fingerprint, page positions, figure dimensions and bibliography count. The [reference audit](evidence/reference_audit.md) records the checked citations and dataset versions.

## Recheck the development splits and matched baselines

From the repository root, run `python -B scripts/audit_dasfaa_splits.py ..` when the original event archive is in the parent directory. The output should equal [split and baseline audit](evidence/split_and_baseline_audit.json). The audit includes per-window fit, validation and main-test counts, hashes of event IDs, the raw SNAP cleaning counts, GCNSI hits on WP1 IDs and paired intervals computed from per-event marginal lists. The main test comes from distilled_source_pairs. The earlier ordinary_source_roles test is a different cohort.

## Reproduce the final transfers and input counts

Run these commands from the project root with the original archive in its parent directory. GCNSI reuses the selected checkpoints, checks the archived Dept2 predictions and writes the three later transfer predictions under runs/dasfaa_gcnsi_transfer.

```text
python -B scripts/evaluate_dasfaa_gcnsi.py
python -B scripts/complete_dasfaa_baselines.py
python -B scripts/audit_dasfaa_top12.py
python -B scripts/dasfaa_archive_counts.py
```

The baseline completion command uses the archived WP3 training recipe for the CollegeMsg Tree and the existing Dept2 Static checkpoints for GNU libc. It verifies the original Dept2 ranks or hits before scoring the sealed WP1 events. The [supplemental baseline record](../runs/dasfaa_baseline_completion/summary.json) links the two sets of ranks to the matching cohorts by SHA-256.
