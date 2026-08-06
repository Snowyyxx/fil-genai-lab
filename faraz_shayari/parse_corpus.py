"""
Parse Ahmad Faraz's Roman-Urdu ghazals into couplet-level records.

Source: amir9ume/urdu_ghazals_rekhta  (en = English transliteration = Roman Urdu),
credited to Rekhta Foundation. Each file is one ghazal; every TWO consecutive
non-empty lines form one sher (couplet), which is the self-contained unit of
meaning we retrieve on.

Output: data/couplets.jsonl  — one JSON object per couplet.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "data" / "faraz_en_ghazals"
OUT = Path(__file__).parent / "data" / "couplets.jsonl"


def title_from_filename(name: str) -> str:
    # strip the "-ahmad-faraz-ghazals" suffix and prettify the slug
    name = re.sub(r"-ahmad-faraz-ghazals$", "", name)
    return name.replace("-", " ").strip()


def main() -> None:
    records = []
    cid = 0
    for path in sorted(SRC.iterdir()):
        if not path.is_file():
            continue
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]                 # drop blank lines
        ghazal = title_from_filename(path.name)
        # pair consecutive lines into couplets (shers)
        for i in range(0, len(lines) - 1, 2):
            line1, line2 = lines[i], lines[i + 1]
            records.append({
                "id": cid,
                "poet": "Ahmad Faraz",
                "ghazal": ghazal,
                "line1": line1,
                "line2": line2,
                "couplet": f"{line1}\n{line2}",
            })
            cid += 1

    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ {len(records)} couplets from {len(list(SRC.iterdir()))} ghazals -> {OUT}")
    print("\nSample couplets:")
    for r in records[:3]:
        print(f"  [{r['id']}] ({r['ghazal'][:40]}…)")
        print(f"      {r['line1']}")
        print(f"      {r['line2']}")


if __name__ == "__main__":
    main()
