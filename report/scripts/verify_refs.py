"""Check every DOI cited in references.bib against Crossref and print the registered metadata.

python report/scripts/verify_refs.py report/references.bib      # or a list of DOIs as arguments
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def lookup(doi: str) -> str:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": "StorePulseReport/1.0 (reference check)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            m = json.load(r)["message"]
    except Exception as exc:  # noqa: BLE001
        return f"NOT FOUND ({exc})"
    authors = m.get("author", [])
    names = ", ".join(f"{a.get('given', '')[:1]}. {a.get('family', '')}" for a in authors[:3])
    names += " et al." if len(authors) > 3 else ""
    date = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0]
    venue = (m.get("container-title") or [""])[0]
    return (
        f"{names} | {m['title'][0][:120]} | {venue[:70]} | vol {m.get('volume', '')} "
        f"no {m.get('issue', '')} pp {m.get('page', '')} | {date[0]}"
    )


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 1 and args[0].endswith(".bib"):
        dois = re.findall(r"doi\s*=\s*\{([^}]+)\}", Path(args[0]).read_text(encoding="utf-8"))
    else:
        dois = args
    for doi in dois:
        print(f"{doi} => {lookup(doi)}", flush=True)


if __name__ == "__main__":
    main()
