"""Conservative data pipeline for the author's declared-fictional CABA demo.

No raw input is written to disk. Only the columns in CANONICAL_COLUMNS survive.
The bundled demo is a minimized projection of the user-authorized source file.
The synthetic generator is retained solely as a reproducible testing fixture.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import unicodedata
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd


MAX_BYTES = 200 * 1024 * 1024
MAX_ROWS = 200_000
CELL_DEGREES = 0.01
LAT_BOUNDS = (-34.71, -34.52)
LON_BOUNDS = (-58.54, -58.32)
REQUIRED_RAW_COLUMNS = (
    "operation", "property_type", "place_name", "state_name", "lat", "lon",
    "price", "currency", "surface_total_in_m2", "surface_covered_in_m2", "rooms",
)
CANONICAL_COLUMNS = (
    "property_type", "neighborhood", "area_total", "area_covered", "rooms",
    "lat", "lon", "price_usd", "usd_m2", "ratio", "spatial_group",
)
PUBLIC_DEMO_COLUMNS = (
    "property_type", "neighborhood", "area_total", "area_covered", "rooms",
    "lat", "lon", "price_usd",
)

# Official neighborhoods plus common real-estate subarea labels. Keeping an
# allowlist prevents a free-text place_name from becoming an address in the UI.
_NEIGHBORHOODS = (
    "Agronomía", "Almagro", "Balvanera", "Barracas", "Belgrano", "Boedo",
    "Caballito", "Chacarita", "Coghlan", "Colegiales", "Constitución",
    "Flores", "Floresta", "La Boca", "La Paternal", "Liniers", "Mataderos",
    "Monte Castro", "Monserrat", "Nueva Pompeya", "Núñez", "Palermo",
    "Parque Avellaneda", "Parque Chacabuco", "Parque Chas", "Parque Patricios",
    "Puerto Madero", "Recoleta", "Retiro", "Saavedra", "San Cristóbal",
    "San Nicolás", "San Telmo", "Vélez Sarsfield", "Versalles", "Villa Crespo",
    "Villa del Parque", "Villa Devoto", "Villa General Mitre", "Villa Lugano",
    "Villa Luro", "Villa Ortúzar", "Villa Pueyrredón", "Villa Real",
    "Villa Riachuelo", "Villa Santa Rita", "Villa Soldati", "Villa Urquiza",
    "Abasto", "Barrio Norte", "Catalinas", "Centro / Microcentro", "Congreso",
    "Las Cañitas", "Once", "Parque Centenario", "Tribunales",
    "Palermo Hollywood", "Palermo Soho", "Palermo Chico", "Palermo Viejo",
)


class DataValidationError(ValueError):
    """An input cannot safely be interpreted as the documented dataset."""


def _normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = " ".join(value.strip().split()).casefold()
    return "".join(c for c in unicodedata.normalize("NFKD", value)
                   if not unicodedata.combining(c))


_NEIGHBORHOOD_MAP = {_normalize(name): name for name in _NEIGHBORHOODS}
_NEIGHBORHOOD_MAP.update({
    "boca": "La Boca", "paternal": "La Paternal", "montserrat": "Monserrat",
    "centro": "Centro / Microcentro", "microcentro": "Centro / Microcentro",
    "centro/microcentro": "Centro / Microcentro",
})


def _read_bytes(source: str | Path | bytes | BinaryIO | TextIO) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataValidationError("No se encontró un archivo legible.")
        if path.stat().st_size > MAX_BYTES:
            raise DataValidationError("El archivo supera el límite de 200 MiB.")
        with path.open("rb") as handle:
            content = handle.read(MAX_BYTES + 1)
    elif isinstance(source, bytes):
        content = source
    elif hasattr(source, "read"):
        old_position = None
        try:
            old_position = source.tell()
            source.seek(0)
        except (AttributeError, OSError, io.UnsupportedOperation):
            pass
        try:
            content = source.read(MAX_BYTES + 1)
        finally:
            if old_position is not None:
                try:
                    source.seek(old_position)
                except (AttributeError, OSError, io.UnsupportedOperation):
                    pass
        if isinstance(content, str):
            content = content.encode("utf-8")
    else:
        raise DataValidationError("Usá un archivo CSV o TSV en UTF-8.")
    if not isinstance(content, bytes):
        raise DataValidationError("No se pudo leer el contenido del archivo.")
    if len(content) > MAX_BYTES:
        raise DataValidationError("El archivo supera el límite de 200 MiB.")
    if not content:
        raise DataValidationError("El archivo está vacío.")
    return content


def load_csv(source: str | Path | bytes | BinaryIO | TextIO) -> tuple[pd.DataFrame, dict]:
    """Read UTF-8 (optional BOM) comma CSV or tab TSV, then validate.

    The input can be a path, bytes, or a Streamlit UploadedFile. There is no
    extension-based delimiter assumption. File objects are rewound and restored.
    Parser errors deliberately omit raw text, which can contain private data.
    """
    try:
        content = _read_bytes(source)
    except (OSError, UnicodeError) as exc:
        raise DataValidationError("No se pudo leer el archivo como UTF-8.") from exc
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise DataValidationError("El archivo debe estar codificado en UTF-8.") from exc
    if "\x00" in text:
        raise DataValidationError("El archivo contiene bytes nulos; exportalo como CSV UTF-8.")
    # csv.reader defaults to only 128 KiB per field. An ignored description can
    # legitimately exceed that. Keep the process-wide parser ceiling at least
    # as large as our already-enforced whole-file byte limit; do not temporarily
    # restore a smaller ceiling while another Streamlit session is parsing.
    if csv.field_size_limit() < MAX_BYTES:
        csv.field_size_limit(MAX_BYTES)
    first_line = text.partition("\n")[0].rstrip("\r")
    choices = []
    for delimiter in (",", "\t"):
        try:
            header = next(csv.reader([first_line], delimiter=delimiter, strict=True))
        except (csv.Error, StopIteration):
            continue
        if set(REQUIRED_RAW_COLUMNS).issubset({column.strip() for column in header}):
            choices.append((delimiter, header))
    if len(choices) != 1:
        raise DataValidationError(
            "No se reconoció el esquema: se requiere CSV separado por comas o TSV, "
            "con las columnas indicadas en el README."
        )
    delimiter, header = choices[0]
    cleaned_header = [column.strip() for column in header]
    if len(cleaned_header) != len(set(cleaned_header)):
        raise DataValidationError("El encabezado contiene columnas duplicadas.")
    # Check shape before pandas reads selected columns: otherwise malformed rows
    # could silently shift a price. Private free-text columns are never loaded
    # into the pandas table. csv.reader visits them only to validate the CSV.
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
        next(reader)
        data_rows = 0
        for row in reader:
            if not row:  # Same blank-line behavior as pandas.
                continue
            data_rows += 1
            if data_rows > MAX_ROWS:
                raise DataValidationError("El archivo supera el límite de 200.000 filas.")
            if len(row) != len(header):
                raise DataValidationError("Hay filas con una cantidad de columnas incorrecta.")
    except csv.Error as exc:
        raise DataValidationError("No se pudo interpretar el CSV/TSV; revisá las comillas.") from exc
    try:
        raw = pd.read_csv(
            io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
            names=cleaned_header, header=0, usecols=list(REQUIRED_RAW_COLUMNS),
            nrows=MAX_ROWS + 1, on_bad_lines="error", index_col=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataValidationError("No se pudo interpretar el CSV/TSV; revisá las comillas y columnas.") from exc
    frame, report = prepare_frame(raw)
    report.update({"delimiter": "tab" if delimiter == "\t" else "comma",
                   "encoding": "utf-8-sig", "file_bytes": len(content)})
    return frame, report


def load_demo(path: str | Path | bytes | BinaryIO | TextIO | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the exact eight-column, minimized public demonstration projection.

    Coordinates must be rounded to three decimals (approximately 100 m here).
    The user describes the supplied source as fictional; this assertion and any
    underlying license are not independently verified by the loader. Constants
    sell/USD/Capital Federal restore the scope already enforced by prepare_demo.
    """
    source = Path(__file__).resolve().parent / "data" / "demo.csv" if path is None else path
    try:
        content = _read_bytes(source)
        text = content.decode("utf-8-sig", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise DataValidationError("No se pudo leer la demo como CSV UTF-8.") from exc
    if "\x00" in text:
        raise DataValidationError("La demo contiene bytes nulos.")
    if csv.field_size_limit() < MAX_BYTES:
        csv.field_size_limit(MAX_BYTES)
    try:
        reader = csv.reader(io.StringIO(text), delimiter=",", strict=True)
        header = next(reader, None)
        if header != list(PUBLIC_DEMO_COLUMNS):
            raise DataValidationError("La demo debe tener exactamente las ocho columnas públicas, en el orden documentado.")
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if count > MAX_ROWS:
                raise DataValidationError("La demo supera el límite de 200.000 filas.")
            if len(row) != len(PUBLIC_DEMO_COLUMNS):
                raise DataValidationError("La demo contiene filas con una cantidad incorrecta de columnas.")
    except csv.Error as exc:
        raise DataValidationError("No se pudo interpretar el CSV de la demo.") from exc
    try:
        projection = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataValidationError("No se pudo interpretar el CSV de la demo.") from exc
    for name in ("lat", "lon"):
        coordinate = pd.to_numeric(projection[name], errors="coerce")
        finite = np.isfinite(coordinate)
        if not np.isclose(coordinate[finite], coordinate[finite].round(3), rtol=0, atol=1e-10).all():
            raise DataValidationError("Las coordenadas de la demo deben estar redondeadas a tres decimales.")
    raw = projection.rename(columns={
        "neighborhood": "place_name", "area_total": "surface_total_in_m2",
        "area_covered": "surface_covered_in_m2", "price_usd": "price",
    }).assign(operation="sell", currency="USD", state_name="Capital Federal")
    frame, report = prepare_frame(raw)
    report.update({"delimiter": "comma", "encoding": "utf-8-sig", "file_bytes": len(content),
                   "source_kind": "minimized_user_authorized_projection", "coordinate_decimals": 3})
    report["notes"].extend([
        "The supplied source is described as fictional by the user; this is not independently verified.",
        "The public projection has exactly eight fields and coordinates rounded to three decimals; location coarsening affects distances and duplicate collisions.",
    ])
    return frame, report


def _spatial_groups(frame: pd.DataFrame) -> pd.Series:
    # Rounding matches the duplicate fingerprint. A 0.01 degree cell is roughly
    # 1.1 km north-south and 0.9 km east-west here; it is not a circular buffer.
    lat_cell = np.floor(frame["lat"].round(5) / CELL_DEGREES + 1e-9).astype("int64")
    lon_cell = np.floor(frame["lon"].round(5) / CELL_DEGREES + 1e-9).astype("int64")
    return lat_cell.astype(str) + ":" + lon_cell.astype(str)


def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fingerprint = frame[["property_type", "rooms", "lat", "lon", "area_total", "area_covered"]].copy()
    fingerprint[["lat", "lon"]] = fingerprint[["lat", "lon"]].round(5)
    fingerprint[["area_total", "area_covered"]] = fingerprint[["area_total", "area_covered"]].round(2)
    duplicate_mask = fingerprint.duplicated(keep="first")
    counts = fingerprint.groupby(list(fingerprint.columns), dropna=False).size()
    with_price = fingerprint.assign(price_usd=frame["price_usd"].to_numpy())
    price_counts = with_price.groupby(list(fingerprint.columns), dropna=False)["price_usd"].nunique()
    return frame.loc[~duplicate_mask].copy().reset_index(drop=True), {
        "duplicate_rows_removed": int(duplicate_mask.sum()),
        "duplicate_feature_groups": int((counts > 1).sum()),
        "duplicate_target_conflict_groups": int((price_counts > 1).sum()),
    }


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the raw Properati schema without retaining free-text fields.

    Rejection counts are sequential and mutually exclusive, in the order below.
    USD price is the original `price` for currency USD; converted prices are not
    used. Missing amenities, bathrooms and timestamps are never inferred.
    """
    if not isinstance(raw, pd.DataFrame):
        raise DataValidationError("Se requiere una tabla de datos.")
    if len(raw) > MAX_ROWS:
        raise DataValidationError("El archivo supera el límite de 200.000 filas.")
    if raw.columns.duplicated().any():
        raise DataValidationError("El encabezado contiene columnas duplicadas.")
    missing = set(REQUIRED_RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError("Faltan columnas requeridas: " + ", ".join(sorted(missing)))
    # Reset index so duplicate external indexes cannot misalign validation masks.
    raw = raw.loc[:, REQUIRED_RAW_COLUMNS].copy().reset_index(drop=True)
    frame = pd.DataFrame(index=raw.index)
    frame["property_type"] = raw["property_type"].map(_normalize).map(
        {"apartment": "apartment", "house": "house", "ph": "PH"})
    frame["neighborhood"] = raw["place_name"].map(_normalize).map(_NEIGHBORHOOD_MAP)
    for target, original in (("area_total", "surface_total_in_m2"),
                             ("area_covered", "surface_covered_in_m2"),
                             ("price_usd", "price"), ("rooms", "rooms"),
                             ("lat", "lat"), ("lon", "lon")):
        frame[target] = pd.to_numeric(raw[original], errors="coerce")
    valid = pd.Series(True, index=raw.index, dtype=bool)
    rejected = {}

    def require(name: str, mask: pd.Series) -> None:
        nonlocal valid
        mask = mask.fillna(False).astype(bool)
        rejected[name] = int((valid & ~mask).sum())
        valid &= mask

    require("operation_not_sell", raw["operation"].map(_normalize).eq("sell"))
    require("unsupported_property_type", frame["property_type"].notna())
    require("outside_declared_caba", raw["state_name"].map(_normalize).eq("capital federal"))
    require("currency_not_usd", raw["currency"].map(_normalize).eq("usd"))
    require("unknown_neighborhood", frame["neighborhood"].notna())
    require("invalid_price", np.isfinite(frame["price_usd"]) & frame["price_usd"].gt(0))
    require("invalid_area", np.isfinite(frame["area_total"]) & frame["area_total"].gt(0)
            & np.isfinite(frame["area_covered"]) & frame["area_covered"].gt(0)
            & frame["area_covered"].le(frame["area_total"]))
    require("invalid_rooms", np.isfinite(frame["rooms"]) & frame["rooms"].between(1, 20)
            & frame["rooms"].eq(frame["rooms"].round()))
    require("invalid_coordinates", np.isfinite(frame["lat"]) & np.isfinite(frame["lon"])
            & frame["lat"].between(*LAT_BOUNDS) & frame["lon"].between(*LON_BOUNDS))
    frame = frame.loc[valid].copy()
    frame["rooms"] = frame["rooms"].astype("int64")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        frame["usd_m2"] = frame["price_usd"] / frame["area_total"]
        frame["ratio"] = frame["area_covered"] / frame["area_total"]
    finite_derived = np.isfinite(frame["usd_m2"]) & frame["usd_m2"].gt(0)
    finite_derived &= np.isfinite(frame["ratio"]) & frame["ratio"].gt(0)
    rejected["invalid_derived_values"] = int((~finite_derived).sum())
    frame = frame.loc[finite_derived].copy()
    frame, duplicate_report = _deduplicate(frame)
    frame["spatial_group"] = _spatial_groups(frame)
    frame = frame.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    report = {
        "input_rows": int(len(raw)), "accepted_rows": int(len(frame)),
        "rejected_rows": int(sum(rejected.values())),
        "rejection_counts": rejected, "rejection_count_method": "sequential_exclusive",
        **duplicate_report,
        "notes": [
            "Only sale listings in declared Capital Federal with original USD prices are retained.",
            "Complete-case filtering can introduce selection bias; excluded listings can differ systematically from retained ones.",
            "Neighborhood names use an allowlist; unknown labels are excluded.",
            "Coordinate bounds are a plausibility rectangle, not the administrative boundary.",
            "Duplicate fingerprint: type, rooms, coordinates rounded to 5 decimals and areas to 2 decimals; price and neighborhood excluded. First valid row retained.",
            "Distinct units sharing these features can collide; repeated listings with changed features can remain.",
            "No reliable listing dates are required by this schema; there is no temporal validation.",
        ],
    }
    return frame, report


def split_data(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deduplicate again, then hold out whole 0.01-degree geographic cells.

    Membership is determined without inspecting target values. Nearby listings
    across a cell boundary can still enter opposite partitions. This measures a
    spatial holdout of the supplied records, not future-price performance.
    """
    if not isinstance(test_fraction, (int, float)) or not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise DataValidationError("La proporción de evaluación debe estar entre 0 y 1.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise DataValidationError("La semilla debe ser un entero no negativo.")
    if not isinstance(df, pd.DataFrame) or not set(CANONICAL_COLUMNS).issubset(df.columns):
        raise DataValidationError("La partición requiere el esquema canónico validado.")
    if df.columns.duplicated().any():
        raise DataValidationError("La tabla tiene columnas duplicadas.")
    safe = df.loc[:, CANONICAL_COLUMNS].copy().reset_index(drop=True)
    numeric = ["area_total", "area_covered", "rooms", "lat", "lon", "price_usd", "usd_m2", "ratio"]
    try:
        if not np.isfinite(safe[numeric].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite")
        safe[numeric] = safe[numeric].astype(float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("La partición requiere valores numéricos finitos.") from exc
    if (safe["area_total"].le(0) | safe["area_covered"].le(0)
            | safe["area_covered"].gt(safe["area_total"]) | safe["price_usd"].le(0)
            | ~safe["rooms"].between(1, 20) | safe["rooms"].ne(safe["rooms"].round())
            | ~safe["lat"].between(*LAT_BOUNDS) | ~safe["lon"].between(*LON_BOUNDS)
            | ~safe["property_type"].isin(["apartment", "house", "PH"])
            | ~safe["neighborhood"].isin(_NEIGHBORHOODS)).any():
        raise DataValidationError("La partición recibió filas sin validar.")
    safe["rooms"] = safe["rooms"].astype("int64")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        safe["usd_m2"] = safe["price_usd"] / safe["area_total"]
        safe["ratio"] = safe["area_covered"] / safe["area_total"]
    derived = safe[["usd_m2", "ratio"]].to_numpy()
    if not np.isfinite(derived).all() or not (derived > 0).all():
        raise DataValidationError("La partición produjo valores derivados inválidos.")
    safe, duplicate_report = _deduplicate(safe)
    safe["spatial_group"] = _spatial_groups(safe)
    groups = np.sort(safe["spatial_group"].unique())
    if len(groups) < 2:
        raise DataValidationError("Se necesitan al menos dos celdas geográficas para separar entrenamiento y evaluación.")
    count = min(len(groups) - 1, max(1, int(np.ceil(len(groups) * test_fraction))))
    test_groups = set(np.random.default_rng(seed).permutation(groups)[:count])
    mask = safe["spatial_group"].isin(test_groups)
    train, test = safe.loc[~mask].copy().reset_index(drop=True), safe.loc[mask].copy().reset_index(drop=True)
    info = {
        "method": "spatial_group_holdout_0.01_degrees", "seed": int(seed),
        "train_rows": len(train), "test_rows": len(test),
        "train_groups": int(train["spatial_group"].nunique()),
        "test_groups": int(test["spatial_group"].nunique()),
        "requested_test_fraction": float(test_fraction),
        "actual_test_fraction": len(test) / len(safe),
        **duplicate_report,
        "notes": [
            "Cells, not individual rows, are sampled; the actual row fraction can differ.",
            "Rounded location/features are deduplicated before splitting, independent of price.",
            "Nearby properties across cell boundaries can remain in opposite partitions.",
            "No temporal holdout is claimed because reliable dates are unavailable.",
        ],
    }
    return train, test, info


def synthetic_data(seed: int = 42, n: int = 600) -> pd.DataFrame:
    """Return invented, reproducible listings as testing fixtures only.

    Coordinates, areas and prices are synthetic. The names of neighborhoods are
    real labels, but locations do not assert any real property or neighborhood
    boundary. Synthetic scores do not measure performance on real listings.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 1 <= n <= MAX_ROWS:
        raise DataValidationError("La cantidad sintética debe ser un entero entre 1 y 200.000.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise DataValidationError("La semilla debe ser un entero no negativo.")
    rng = np.random.default_rng(seed)
    # Each label spreads across several grid cells, so the default spatial split
    # can evaluate neighborhoods also observed during training.
    centers = [(-34.580, -58.420), (-34.598, -58.402), (-34.620, -58.447), (-34.560, -58.457)]
    names = np.array(["Palermo", "Recoleta", "Caballito", "Belgrano"])
    neighborhood_ids = np.arange(n) % len(names)
    rng.shuffle(neighborhood_ids)
    lat = np.array([centers[i][0] for i in neighborhood_ids]) + rng.uniform(-0.015, 0.015, n)
    lon = np.array([centers[i][1] for i in neighborhood_ids]) + rng.uniform(-0.018, 0.018, n)
    types = rng.choice(["apartment", "house", "PH"], n, p=[0.65, 0.15, 0.20])
    rooms = rng.integers(1, 6, n)
    total = np.round(18 + rooms * 20 + rng.uniform(0, 35, n), 2)
    covered = np.round(total * rng.uniform(0.64, 1.0, n), 2)
    invented_rate = 1300 + np.array([330, 450, 0, 260])[neighborhood_ids]
    invented_rate = invented_rate + 140 * (types == "apartment") + rng.normal(0, 130, n)
    price = np.round(np.maximum(100, invented_rate) * total + rng.normal(0, 3500, n), 2)
    raw = pd.DataFrame({
        "operation": "sell", "property_type": types, "place_name": names[neighborhood_ids],
        "state_name": "Capital Federal", "currency": "USD", "price": np.maximum(1000, price),
        "surface_total_in_m2": total, "surface_covered_in_m2": covered,
        "rooms": rooms, "lat": lat, "lon": lon,
    })
    frame, _ = prepare_frame(raw)
    return frame
