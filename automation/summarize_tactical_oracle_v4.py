from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from automation.evaluate_tactical_oracle_v4 import summarize_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed Tactical Oracle v4 shards")
    parser.add_argument("--oracle-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Oracle aggregate: {output}")
    records = []
    roots = [path.resolve() for path in args.oracle_root]
    event_sources: dict[str, str] = {}
    for root in roots:
        pairs_path = root / "pairs.json"
        oracle_path = root / "oracle.json"
        if not pairs_path.is_file() or not oracle_path.is_file():
            raise ValueError(f"incomplete Tactical Oracle root: {root}")
        for row in json.loads(pairs_path.read_text(encoding="utf-8")):
            previous = event_sources.setdefault(row["event_id"], str(root))
            if previous != str(root):
                raise ValueError(f"duplicate event across Oracle roots: {row['event_id']}")
            records.append(row)
    summary = summarize_oracle(records)
    summary.update(
        {
            "schema_version": "tactical_oracle_v4.aggregate.v1",
            "source_roots": [
                str(path.relative_to(ROOT)).replace("\\", "/") for path in roots
            ],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "oracle"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
