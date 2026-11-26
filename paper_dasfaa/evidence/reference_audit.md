# Reference audit

The bibliography cites 39 records in the current manuscript. The [inherited reference audit](../../../归档/DASFAA2027_定向修订/paper_review/reference_audit.md) covers their identities and sources. The revised build uses the following bibliographic updates.

| PDF entry | Update | Record |
| --- | --- | --- |
| 1 | Added the publisher URL | [PMLR, Alexandari et al.](https://proceedings.mlr.press/v119/alexandari20a.html) |
| 5 | Formatted the third author as Nguyen, Q.V.H. | [ACM paper](https://doi.org/10.1145/3357384.3357994) and [Griffith University author profile](https://experts.griffith.edu.au/9203-henry-nguyen) |
| 7 | Added the publisher URL | [NeurIPS Proceedings, Garg et al.](https://proceedings.neurips.cc/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html) |
| 15 | Shortened the venue to Proceedings of STOC | [ACM proceedings paper](https://doi.org/10.1145/335305.335364) |
| 38 | Shortened the venue to Proceedings of AAAI | [AAAI proceedings paper](https://doi.org/10.1609/aaai.v38i20.30262) |

## Proofreading checks on 26 November 2026

- Confirmed the identifier, authors, title, and 2026 submission year of Brack et al. [arXiv record](https://arxiv.org/abs/2611.23175)
- Verified the Network Repository dataset name with hyphens and the live URL with underscores. [Manufacturing email dataset](https://networkrepository.com/ia_radoslaw_email.php)
- Confirmed the Enron data documentation at revision d61dabac. [Pinned revision](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/man/enron.Rd)
- Confirmed article 068702 and the DOI of Pinto et al. [Physical Review Letters](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.068702)
- Added the dataset distribution entry alongside the original CollegeMsg research paper. [SNAP CollegeMsg](https://snap.stanford.edu/data/CollegeMsg.html)

The final check uses the publisher's DOI spelling for Pinto et al. and keeps the complete DOI on one PDF line. The Company reference's linked address was reopened and returned HTTP 200. [APS article](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.068702) · [Network Repository](https://networkrepository.com/ia_radoslaw_email.php)

## Enron software citation and input provenance

CRAN identifies Gabor Csardi as the author of igraphdata 1.0.1 and records its publication on 13 July 2015. The package reference appears as [5] in the current PDF. [CRAN package record](https://cran.r-project.org/package=igraphdata) · [Enron manual entry](https://cran.r-project.org/web/packages/igraphdata/refman/igraphdata.html#enron)

The archived experimental input matches the GitHub enron.rda file at d61dabac88607cf726323dcc68109fb5753aebc7, SHA-256 68df616583a009c323131574c833f615443756087dda9a97e27a0ad8d658fd5a. The tree reports development version 1.0.1.9019. The pinned commit updates the roxygen2 configuration version in DESCRIPTION. The package source on CRAN ships a file with SHA-256 f07453c8f09e1ecac2bf6a1fff9477b4d4b854308e74b950d50c069cc8c1b286. [Commit diff](https://github.com/igraph/igraphdata/commit/d61dabac88607cf726323dcc68109fb5753aebc7) · [Development DESCRIPTION](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/DESCRIPTION) · [Pinned Enron file](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/data/enron.rda)
