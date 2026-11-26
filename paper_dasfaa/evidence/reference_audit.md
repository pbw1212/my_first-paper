# Reference audit

The [current bibliography](../references.bib) contains 39 records. The table records selected checks by BibTeX key so that the entries can be traced across revisions.

| BibTeX key | Update | Record |
| --- | --- | --- |
| `alex` | Added the publisher URL | [PMLR, Alexandari et al.](https://proceedings.mlr.press/v119/alexandari20a.html) |
| `gcnsi` | Formatted the third author as Nguyen, Q.V.H. | [ACM paper](https://doi.org/10.1145/3357384.3357994) and [Griffith University author profile](https://experts.griffith.edu.au/9203-henry-nguyen) |
| `garg` | Added the publisher URL | [NeurIPS Proceedings, Garg et al.](https://proceedings.neurips.cc/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html) |
| `kempe_j` | Cited the journal version | [Journal of Computer and System Sciences](https://doi.org/10.1006/jcss.2002.1829) |
| `bosoul` | Shortened the venue to Proceedings of AAAI | [AAAI proceedings paper](https://doi.org/10.1609/aaai.v38i20.30262) |

## Proofreading checks on 26 November 2026

- Confirmed the identifier, authors, title, and 2026 submission year of Brack et al. [arXiv record](https://arxiv.org/abs/2611.23175)
- Verified the Network Repository dataset name with hyphens and the live URL with underscores. [Manufacturing email dataset](https://networkrepository.com/ia_radoslaw_email.php)
- Confirmed the Enron data documentation at revision d61dabac. [Pinned revision](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/man/enron.Rd)
- Confirmed article 068702 and the DOI of Pinto et al. [Physical Review Letters](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.068702)
- Added the dataset distribution entry alongside the original CollegeMsg research paper. [SNAP CollegeMsg](https://snap.stanford.edu/data/CollegeMsg.html)

The final check uses the publisher's DOI spelling for Pinto et al. and keeps the complete DOI on one PDF line. The Company reference's linked address was reopened and returned HTTP 200. [APS article](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.068702) · [Network Repository](https://networkrepository.com/ia_radoslaw_email.php)

## Enron development snapshot and input provenance

The experiments used the Enron data file at commit d61dabac88607cf726323dcc68109fb5753aebc7 (10 October 2026). The bibliography cites igraphdata development version 1.0.1.9019 and credits the three authors listed with the aut role in DESCRIPTION. [Commit](https://github.com/igraph/igraphdata/commit/d61dabac88607cf726323dcc68109fb5753aebc7) · [Pinned DESCRIPTION](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/DESCRIPTION)

The commit changes the roxygen2 configuration line in DESCRIPTION. It identifies the repository snapshot from which the experiment obtained data/enron.rda. The locally archived input and the downloaded snapshot both hash to SHA-256 68df616583a009c323131574c833f615443756087dda9a97e27a0ad8d658fd5a. The derived contacts archive hashes to 17cc2ea27a871a8046e503dd17f6f3cefadf65d6e924f7fd392f811c8158e14e, matching the experiment provenance record. The CRAN 1.0.1 tarball contains an enron.rda with SHA-256 f07453c8f09e1ecac2bf6a1fff9477b4d4b854308e74b950d50c069cc8c1b286. [Pinned data file](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/data/enron.rda) · [Experiment provenance](../../runs/temporal_transfer/Enron/provenance.json) · [CRAN package record](https://cran.r-project.org/package=igraphdata)
