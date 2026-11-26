"""Check an LNCS build, copy its PDF to the manuscript, and record the checked properties."""
import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import fitz

from verify_dasfaa_tables import PAPER, table


def compact(value):
    s = unicodedata.normalize("NFKC", value).replace(r"\dagger", "†").replace("---", "—").replace("--", "–")
    s = re.sub(r"\\multicolumn\{2\}\{c\}\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\singlefitstar", "*")
    for token in (r"\textbf", r"\underline", r"\boldmath", r"\text{", r"\;", r"\,", "$", "{", "}", "^"):
        s = s.replace(token, "")
    return re.sub(r"\s+", "", s).replace("−", "-").replace("∗", "*").casefold()


def main(folder):
    aux = (folder / "main.aux").read_text(errors="replace")
    log = (folder / "main.log").read_text(errors="replace")
    source = (PAPER / "main.tex").read_text(encoding="utf8").replace(
        r"\input{evaluation}", (PAPER / "evaluation.tex").read_text(encoding="utf8"))
    doc = fitz.open(folder / "main.pdf")
    pages = [page.get_text() for page in doc]
    assert len(pages) == 16
    references_page, = [number for number, page in enumerate(pages, 1)
                        if "References" in page.splitlines()]
    assert references_page == 14 and re.search(r"(?m)^1\. ", pages[references_page - 1])
    assert all(abs(page.rect.width - 595.276) < .02 and abs(page.rect.height - 841.890) < .02 for page in doc)
    assert r"\author{Anonymous Author(s)}" in source and r"\institute{Anonymous Institution}" in source
    assert doc.metadata["author"] == "Anonymous Author(s)"
    assert doc.metadata["title"] == "Recovering Both Diffusion Sources from a Late Cascade Suffix"
    assert not doc.get_xml_metadata() and doc.embfile_count() == 0
    links = [link for page in doc for link in page.get_links()]
    assert all(doc.xref_get_key(link["xref"], "A/S")[1] in ("/GoTo", "/URI") for link in links)
    assert all(link["uri"].startswith(("https://", "http://")) for link in links if link["kind"] == fitz.LINK_URI)
    bibliography = (PAPER / "references.bib").read_text(encoding="utf8")
    expected_urls = set(re.findall(r"\burl=\{([^}]+)\}", bibliography))
    expected_dois = {"https://doi.org/" + doi for doi in re.findall(r"\bdoi=\{([^}]+)\}", bibliography)}
    reference_links = expected_urls | expected_dois
    assert len(expected_urls) == 15 and len(expected_dois) == 24
    assert {link["uri"] for link in links if link["kind"] == fitz.LINK_URI} == reference_links
    reference_text = re.sub(r"\s+", "", "".join(pages[references_page - 1:]))
    assert all(url in reference_text for url in reference_links)
    assert any("https://doi.org/10.1103/PhysRevLett.109.068702" in page
               for page in pages[references_page - 1:])
    inspectable = json.dumps([doc.metadata, doc.get_toc(), links, pages], ensure_ascii=False, default=str)
    assert not re.search(r"(?<![a-z])[a-z]:[\\/]|file:/+|/Users/|/home/", inspectable, re.I)
    assert not re.search(r"\bDO\b|S_i|\\appendix", source)
    assert re.findall(r"\\author\{([^}]+)\}", source) == ["Anonymous Author(s)"]
    assert re.findall(r"\\institute\{([^}]+)\}", source) == ["Anonymous Institution"]
    assert "??" not in "".join(pages) and "github.com/" not in source
    assert r"\(" not in source and r"\)" not in source
    assert not re.search(r"(?<!#)\d%", source), "Escape percentage signs after numbers in manuscript text."
    assert "Both-in-12" not in source
    assert not re.search(r"^!|Overfull \\[hv]box|There were undefined|Reference .* undefined|Citation .* undefined",
                         log, flags=re.M)
    page_map = {}
    for prefix, count in (("f", 6), ("t", 6)):
        positions, mentions = [], []
        for i in range(1, count + 1):
            match = re.search(r"\\newlabel\{" + prefix + str(i) + r"\}\{\{(\d+)\}\{(\d+)\}", aux)
            assert match and int(match[1]) == i
            positions.append(int(match[2]))
            mentions.append(source.index(r"\ref{" + prefix + str(i) + "}")
                            if r"\ref{" + prefix + str(i) + "}" in source
                            else source.index(("\\" + ("figref" if prefix == "f" else "tabref")) +
                                              "{" + prefix + str(i) + "}"))
        assert positions == sorted(positions) and mentions == sorted(mentions)
        page_map[prefix] = positions

    rendered = {}
    for n in range(2, 7):
        page = compact(pages[page_map["t"][n-1]-1])
        for row in table(n):
            assert compact("".join(row)) in page, (n, row)
        rendered[str(n)] = len(table(n))
    top12 = compact(pages[page_map["t"][2]-1])
    for metric in ("Both@12", "Any@12"):
        assert compact(f"{metric} Network Random GCNSI") in top12, (3, metric)
    assert all(compact(label) in top12 for label in ("MLP recall", "Mean pool"))
    heldout_page = pages[page_map["t"][4] - 1]
    assert any("r-package-devel2021–22" in compact(line) for line in heldout_page.splitlines())
    header_page = compact(heldout_page)
    for label in ("Role-Full (%)", "Role-Full - Posts", "pp [95% CI]"):
        assert compact(label) in header_page, ("Table 5 header", label)
    first = compact(pages[page_map["t"][0]-1])
    for group in ("Personal 26 Activity (10):", "Visibility variables (2):", "Shared summaries (4):",
                  "+ Recent contacts 34 Direct ties to suffix (4):", "+ Full temporal history 64 Multi-hop (6):",
                  "Replay (7):", "Replay summaries (2):", "two tied visibility ranks"):
        assert compact(group) in first, group
    assert "top-K list" in pages[0]
    assert sum(page.count("f\u0304_m") for page in pages) == 2
    depth_caption = pages[page_map["t"][5] - 1].split("Table 6.", 1)[1].split("Unscored", 1)[0]
    assert all(depth in depth_caption for depth in ("D_0", "D_25", "D_50"))
    assert r"\newlabel{prop:oracle}{{1}" in aux and r"\newlabel{sec:controlled_role}{{6.3}" in aux
    equations = re.findall(r"\\begin{equation}.*?\\label\{([^}]+)\}", source, re.S)
    assert equations == ["eq:normalizer", "oracle"]
    for label in equations:
        assert rf"\eqref{{{label}}}" in source and rf"\newlabel{{{label}}}" in aux

    figures = {}
    labels = {
        "fig1_overview": ("Unobserved transmission", "Pre-window tie", "Personal 26",
                          "+ Recent contacts 34", "+ Full temporal history 64", "Top-K candidate list"),
        "fig2_budgets": ("List size", "Budget (% of legal pool)", "Joint coverage (%)"),
        "fig3_selectivity": ("Simulated: vs. personal history", "Mailing lists: vs. post volume", "Both@12 gain (pp)",
                              "CollegeMsg", "Company", "Python-Dev", "Python-Ideas", "R-package-devel",
                              "GNU libc 2021–22", "GNU libc 2024"),
        "fig4_prior_and_ceiling": ("Dept2", "Company", "Dept4", "Enron", "80% prod.", "Oracle ceiling (%)"),
        "fig5_multisource": ("All sources", "Per-source recall", "Actual source count"),
        "fig6_sim2real_decomp": ("Sender activity prior", "First-recipient target"),
    }
    for path in sorted((PAPER / "figures").glob("*.pdf")):
        with fitz.open(path) as fig:
            pg = fig[0]
            assert len(fig) == 1 and abs(pg.rect.width - 4.8 * 72) < 0.01
            assert not pg.get_images() and pg.get_drawings() and pg.get_fonts()
            font_sizes = [span["size"] for block in pg.get_text("dict")["blocks"]
                          for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            assert min(font_sizes) >= 6, (path.name, min(font_sizes))
            fig_text = compact(pg.get_text())
            for label in labels[path.stem]:
                assert compact(label) in fig_text, (path.name, label)
            if path.stem == "fig5_multisource":
                labels_at_three = [span for block in pg.get_text("dict")["blocks"]
                                   for line in block["lines"] for span in line["spans"]
                                   if span["text"] in {"89.66", "89.82"}]
                assert len(labels_at_three) == 2
                assert abs(labels_at_three[0]["origin"][1] - labels_at_three[1]["origin"][1]) < 0.01
            if "sim2real_decomp" in path.name:
                depths = []
                for block in pg.get_text("dict")["blocks"]:
                    for line in block["lines"]:
                        spans = line["spans"]
                        for base, sub in zip(spans, spans[1:]):
                            if base["text"] == "D":
                                assert sub["size"] < base["size"] and sub["origin"][1] > base["origin"][1]
                                depths.append(sub["text"])
                assert sorted(depths) == ["0", "0", "50", "50"]
            figures[path.name] = dict(width_inches=4.8, height_inches=round(pg.rect.height/72, 2),
                                      min_font_pt=round(min(font_sizes), 2), raster_images=0)
    assert len(figures) == len(labels)
    related = source.split(r"\section{Related Work}")[1].split(r"\section{Retrieval")[0]
    cited = {k for values in re.findall(r"\\cite\{([^}]+)\}", related) for k in values.split(",")}
    entries = (folder / "main.bbl").read_text(encoding="utf8", errors="replace").count(r"\bibitem")
    assert 25 <= len(cited) <= 30 and entries == 39
    citation_groups = re.findall(r"\\citation\{([^}]+)\}", aux)
    numbers = {k: int(n) for k, n in re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux)}
    assert {k for group in citation_groups for k in group.split(",")} == set(numbers)
    for group in citation_groups:
        seq = [numbers[k] for k in group.split(",")]
        assert seq == sorted(seq), (group, seq)
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\keywords", 1)[0]
    assert len(abstract.split()) <= 200
    assert r"$(\mathrm{rank}_i-0.5)/|V|$" in source and r"\pi(\{a,b\})" in source
    assert r"\sqrt{w_iw_j}" in source
    archive_range = f"[{numbers['pythondev']}–{numbers['rpackagedevel']},{numbers['sambatechnical']}]"
    assert archive_range in compact("".join(pages)), archive_range
    intro = compact("".join(pages[:3]))
    assert all(phrase in intro for phrase in map(compact, (
        "1. We derive a linear-time pair normalizer",
        "2. An exact complete-trace posterior",
        "3. Historical speaking roles improve joint coverage")))
    pdf = folder / "main.pdf"
    target = PAPER / "main.pdf"
    report = dict(pages_including_references=len(doc), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  paper_size_mm=[210,297], anonymous_pdf_author=doc.metadata["author"], latex_errors=0, overfull_horizontal_boxes=0,
                  undefined_references=0, anonymous_metadata_checked=True, embedded_files=0,
                  figure_pages=page_map["f"], table_pages=page_map["t"],
                  table_data_rows_matched_in_compiled_pdf=rendered, related_work_references=len(cited),
                  bibliography_entries=entries, bibliography_starts_on_page=references_page,
                  bibliography_full_url_count=len(expected_urls), bibliography_doi_count=len(expected_dois),
                  abstract_words=len(abstract.split()), figures=figures)
    doc.close()
    if pdf.resolve() != target.resolve():
        shutil.copyfile(pdf, target)
    (PAPER / "evidence/build_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(f"{len(pages)} pages, six tables, six vector figures and {entries} references verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_directory", type=Path, nargs="?", default=PAPER)
    main(parser.parse_args().build_directory)
