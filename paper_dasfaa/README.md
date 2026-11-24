# Recovering Both Diffusion Sources from a Late Cascade Suffix

The anonymous manuscript is [main.pdf](main.pdf). The entry point is [main.tex](main.tex), with experimental sections in [evaluation.tex](evaluation.tex). The project includes the LNCS class, the Springer bibliography style, [references.bib](references.bib), seven vector PDFs, and six generated LaTeX tables.

## Rebuild

From this directory, run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The included figures and tables suffice to build the paper without the experimental archive.

From the repository root, the following commands regenerate the exhibits and check the compiled manuscript. The final command takes an optional directory containing the LaTeX log, bibliography, auxiliary file and PDF when building out of tree.

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
| Table 2 | [Inventory](evidence/inventory.json) records distinct nodes or posting addresses, cleaned contacts or messages, and periods. The archive does not supply all of those counts in one summary. With the original input archive available to the repository readers, dasfaa_archive_counts.py recounts them. Evaluation windows and eligible counts come from WP1, WP2 and WP5. |
| Table 3 | WP1 supplies budget and rank metrics. WP3 supplies tree and FGSSI results. Factorial history and the frozen Dept4 report supply static comparisons. Two unavailable comparisons are marked with a dash. CollegeMsg's archived static and FGSSI values use one seed. |
| Table 4 | WP3 stress results provide the rule-specific and common-draw populations, paired means and both interval endpoints. |
| Table 5 | The heldout_r_lists/early_only/role_evidence/closure summary records 1,793 D0 threads. Each method column counts joint hits. The one-visible-member threshold applies here. |
| Table 6 | The WP5 real/common results hold the discussion IDs fixed across all three depths. The full model is the frozen temporal MLP. The three-visible-member scoring threshold applies at every depth. |
| Figs. 1–3 | Fig. 1 is a schematic of the observation protocol. WP1 and WP2 supply the numerical budget and selectivity panels. |
| Fig. 4 | trained_source_mixtures supplies the empirical means. Its summary contains paired contrast intervals. Marginal method intervals are reconstructed from archived Enron hits using the original seed and three-window block procedure. [Window totals](evidence/prior_window_hits.json) permit regeneration without the event archive. The reconstruction also reproduces the archived paired intervals. source_prior_shift supplies the separate complete-trace ceiling components. |
| Figs. 5–7 | WP4 supplies unknown-count strata. WP5 supplies fixed-thread depth curves and the simulated prior and role interventions on common draws. |

[Table audit](evidence/table_audit.json) records the row and cell checks. [Compiled PDF checks](evidence/build_check.json) record the manuscript fingerprint, page positions, figure dimensions and bibliography count. The references file includes commented candidate entries marked TODO_VERIFY, excluded from the paper's citations.
