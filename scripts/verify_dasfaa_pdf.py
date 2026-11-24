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
    s = unicodedata.normalize("NFKC", value).replace("---", "—").replace("--", "–")
    for token in (r"\(", r"\)", r"\;", "{", "}", "^"):
        s = s.replace(token, "")
    return re.sub(r"\s+", "", s).replace("−", "-").replace("∗", "*").casefold()


def main(folder):
    aux = (folder / "main.aux").read_text(errors="replace")
    log = (folder / "main.log").read_text(errors="replace")
    source = (PAPER / "main.tex").read_text(encoding="utf8").replace(
        r"\input{evaluation}", (PAPER / "evaluation.tex").read_text(encoding="utf8"))
    doc = fitz.open(folder / "main.pdf")
    pages = [page.get_text() for page in doc]
    assert 15 <= len(pages) <= 16
    assert r"\author{Anonymous Author(s)}" in source and r"\institute{Anonymous Institution}" in source
    assert doc.metadata["author"] == "Anonymous Author(s)"
    assert not re.search(r"\bDO\b|S_i|\\appendix", source)
    assert re.findall(r"\\author\{([^}]+)\}", source) == ["Anonymous Author(s)"]
    assert re.findall(r"\\institute\{([^}]+)\}", source) == ["Anonymous Institution"]
    assert "??" not in "".join(pages) and "github.com/" not in source
    assert not re.search(r"^!|Overfull \\[hv]box|There were undefined|Reference .* undefined|Citation .* undefined",
                         log, flags=re.M)
    page_map = {}
    for prefix, count in (("f", 7), ("t", 6)):
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
    first = compact(pages[page_map["t"][0]-1])
    for number in ("26", "34", "64", "0–5", "19–22", "23–25", "6–11", "13–16", "26–28"):
        assert compact(number) in first

    figures = {}
    for path in sorted((PAPER / "figures").glob("*.pdf")):
        with fitz.open(path) as fig:
            pg = fig[0]
            assert len(fig) == 1 and abs(pg.rect.width - 4.8 * 72) < 0.01
            assert not pg.get_images() and pg.get_drawings() and pg.get_fonts()
            if "real_depth" in path.name:
                labels = pg.get_text().splitlines()
                assert all(labels.count(d) == 6 for d in ("D0", "D25", "D50"))
                assert all(m in labels for m in ("Post volume", "Role history", "Roles + replies", "Full temporal"))
            figures[path.name] = dict(width_inches=4.8, height_inches=round(pg.rect.height/72, 2), raster_images=0)
    assert len(figures) == 7
    related = source.split(r"\section{Related Work}")[1].split(r"\section{Retrieval")[0]
    cited = {k for values in re.findall(r"\\cite\{([^}]+)\}", related) for k in values.split(",")}
    entries = (folder / "main.bbl").read_text(encoding="utf8", errors="replace").count(r"\bibitem")
    assert 25 <= len(cited) <= 30 and entries >= len(cited)
    pdf = folder / "main.pdf"
    target = PAPER / "main.pdf"
    report = dict(pages_including_references=len(doc), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  anonymous_pdf_author=doc.metadata["author"], latex_errors=0, overfull_horizontal_boxes=0,
                  undefined_references=0, figure_pages=page_map["f"], table_pages=page_map["t"],
                  table_data_rows_matched_in_compiled_pdf=rendered, related_work_references=len(cited),
                  bibliography_entries=entries, figures=figures)
    doc.close()
    if pdf.resolve() != target.resolve():
        shutil.copyfile(pdf, target)
    (PAPER / "evidence/build_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(f"{len(pages)} pages, six tables, seven vector figures and {entries} references verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_directory", type=Path, nargs="?", default=PAPER)
    main(parser.parse_args().build_directory)
