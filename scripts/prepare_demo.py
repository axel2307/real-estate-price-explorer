"""Rebuild the authorized public projection without changing its source file.

Run from the repository root:
    python scripts/prepare_demo.py --source PATH_TO_AUTHORIZED_CSV

Only run this script on a source whose publication rights and privacy have been
reviewed. It does not verify authorship, fictional status or an upstream license.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data import PUBLIC_DEMO_COLUMNS, DataValidationError, load_csv, load_demo


def prepare_demo(source: str | Path, output: str | Path) -> dict:
    """Write only reviewed columns, then round, revalidate and rededuplicate.

    The first valid row is retained for any collisions created by coordinate
    coarsening. Prices and areas are not generated, imputed or perturbed.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or (output.exists() and source.exists() and source.samefile(output)):
        raise DataValidationError("El destino no puede ser el archivo original.")
    frame, source_report = load_csv(source)
    projection = frame.loc[:, PUBLIC_DEMO_COLUMNS].copy()
    projection[["lat", "lon"]] = projection[["lat", "lon"]].round(3)
    # load_demo restores the fixed scope, applies the same validation, and
    # removes newly coincident feature fingerprints after rounding locations.
    rounded, rounded_report = load_demo(io.StringIO(projection.to_csv(index=False)))
    if rounded.empty:
        raise DataValidationError("No quedaron filas válidas para la demo pública.")
    public = rounded.loc[:, PUBLIC_DEMO_COLUMNS].copy()
    for coordinate in ("lat", "lon"):
        public[coordinate] = public[coordinate].map(lambda value: f"{value:.3f}")
    csv_bytes = public.to_csv(index=False, lineterminator="\n").encode("utf-8")
    verified, verification_report = load_demo(csv_bytes)
    if len(verified) != len(public) or verification_report["rejected_rows"] or verification_report["duplicate_rows_removed"]:
        raise DataValidationError("La verificación final de la demo no conservó todas las filas.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(csv_bytes)
    return {
        "source_rows": source_report["input_rows"],
        "source_valid_rows": source_report["accepted_rows"],
        "source_duplicate_rows_removed": source_report["duplicate_rows_removed"],
        "rounding_rejected_rows": rounded_report["rejected_rows"],
        "rounding_duplicate_rows_removed": rounded_report["duplicate_rows_removed"],
        "rounding_duplicate_feature_groups": rounded_report["duplicate_feature_groups"],
        "rounding_duplicate_target_conflict_groups": rounded_report["duplicate_target_conflict_groups"],
        "public_rows": len(public),
        "public_columns": list(PUBLIC_DEMO_COLUMNS),
        "coordinate_decimals": 3,
        "file_bytes": len(csv_bytes),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "notes": [
            "The user describes the authorized source as fictional; that description is not independently verified.",
            "Only the eight allowlisted fields are exported; all raw addresses, URLs, titles, descriptions, contacts and identifiers are excluded.",
            "Coordinates are rounded to three decimals, approximately 100 meters here; this minimization is not a claim of anonymization.",
            "Coordinate coarsening can merge distinct units with matching features; the first valid record is kept.",
            "Prices, areas, rooms, property types and approved neighborhood labels come from the validated source; no replacement observations are generated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Reviewed and authorized source CSV/TSV")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "data" / "demo.csv")
    args = parser.parse_args()
    try:
        report = prepare_demo(args.source, args.output)
    except (DataValidationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
