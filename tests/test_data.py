import io
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

import data
from data import DataValidationError, load_csv, load_demo, prepare_frame, split_data, synthetic_data


def raw_frame(n=3):
    return pd.DataFrame({
        "operation": ["sell"] * n, "property_type": ["apartment"] * n,
        "place_name": ["Palermo"] * n, "state_name": ["Capital Federal"] * n,
        "currency": ["USD"] * n, "price": [100_000] * n,
        "surface_total_in_m2": [50] * n, "surface_covered_in_m2": [45] * n,
        "rooms": [2] * n, "lat": [-34.58 - i * 0.02 for i in range(n)],
        "lon": [-58.42] * n,
        "description": ["PRIVATE TEXT"] * n, "address": ["PRIVATE ADDRESS"] * n,
    })


@pytest.mark.parametrize("separator", [",", "\t"])
def test_csv_and_tsv_with_bom(separator):
    source = io.BytesIO(raw_frame().to_csv(index=False, sep=separator).encode("utf-8-sig"))
    source.seek(8)
    frame, report = load_csv(source)
    assert len(frame) == 3
    assert source.tell() == 8
    assert tuple(frame.columns) == data.CANONICAL_COLUMNS
    assert "PRIVATE" not in frame.to_csv(index=False)
    assert report["delimiter"] == ("comma" if separator == "," else "tab")
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("column,value,reason", [
    ("surface_total_in_m2", 0, "invalid_area"),
    ("surface_covered_in_m2", 51, "invalid_area"),
    ("surface_covered_in_m2", 0, "invalid_area"),
    ("price", -2, "invalid_price"), ("price", np.inf, "invalid_price"),
    ("rooms", 2.5, "invalid_rooms"), ("rooms", 0, "invalid_rooms"),
    ("rooms", 21, "invalid_rooms"), ("rooms", np.nan, "invalid_rooms"),
    ("lat", 0, "invalid_coordinates"), ("lon", np.inf, "invalid_coordinates"),
    ("currency", "ARS", "currency_not_usd"), ("operation", "rent", "operation_not_sell"),
    ("property_type", "store", "unsupported_property_type"),
    ("state_name", "Buenos Aires", "outside_declared_caba"),
    ("place_name", "A private street 123", "unknown_neighborhood"),
])
def test_reject_invalid_values(column, value, reason):
    raw = raw_frame()
    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = value
    frame, report = prepare_frame(raw)
    assert len(frame) == 2
    assert report["rejection_counts"][reason] == 1
    assert report["input_rows"] == report["accepted_rows"] + report["rejected_rows"] + report["duplicate_rows_removed"]


def test_uses_original_usd_price_and_canonical_neighborhood():
    raw = raw_frame(1)
    raw["price_aprox_usd"] = 1
    raw["place_name"] = "  NUÑEZ  "
    frame, _ = prepare_frame(raw)
    assert frame.iloc[0]["price_usd"] == 100_000
    assert frame.iloc[0]["usd_m2"] == 2_000
    assert frame.iloc[0]["ratio"] == .9
    assert frame.iloc[0]["neighborhood"] == "Núñez"


def test_price_and_neighborhood_independent_dedup_and_disjoint_split():
    raw = raw_frame(3)
    dupe = raw.iloc[[0]].copy()
    dupe["price"] = 130_000
    dupe["place_name"] = "Barrio Norte"
    dupe["lat"] += 0.0000001
    raw = pd.concat([raw, dupe], ignore_index=True)
    frame, report = prepare_frame(raw)
    assert len(frame) == 3
    assert report["duplicate_rows_removed"] == 1
    assert report["duplicate_feature_groups"] == 1
    assert report["duplicate_target_conflict_groups"] == 1
    train, test, info = split_data(pd.concat([frame, frame.iloc[[0]]]))
    assert len(train) + len(test) == 3
    assert set(train.spatial_group).isdisjoint(test.spatial_group)
    assert info["duplicate_rows_removed"] == 1
    assert train.spatial_group.nunique() and test.spatial_group.nunique()
    json.dumps(info, allow_nan=False)


def test_split_ignores_supplied_group_and_target_values():
    frame = synthetic_data()
    train_a, test_a, _ = split_data(frame)
    altered = frame.copy()
    altered["spatial_group"] = "malicious-same-cell"
    altered["price_usd"] *= 2
    _, test_b, _ = split_data(altered)
    assert test_a[["lat", "lon"]].equals(test_b[["lat", "lon"]])
    assert set(train_a.spatial_group).isdisjoint(test_a.spatial_group)


def test_synthetic_is_reproducible_valid_and_spatially_diverse():
    frame = synthetic_data()
    assert len(frame) == 600
    assert frame.equals(synthetic_data())
    assert not frame.equals(synthetic_data(seed=7))
    train, test, info = split_data(frame)
    assert set(train.neighborhood) == set(test.neighborhood)
    assert info["train_groups"] > 5 and info["test_groups"] > 5
    assert np.isfinite(frame.select_dtypes("number")).all().all()
    assert frame["area_total"].ge(frame["area_covered"]).all()


def test_empty_valid_input_returns_report_but_cannot_split():
    frame, report = prepare_frame(raw_frame(0))
    assert frame.empty
    assert report["accepted_rows"] == 0
    with pytest.raises(DataValidationError, match="dos celdas"):
        split_data(frame)


@pytest.mark.parametrize("payload", [b"", b"foo;bar\n1;2\n", b"\xff", b"name,price\nthing,2\n"])
def test_bad_files_have_explicit_errors(payload):
    with pytest.raises(DataValidationError):
        load_csv(payload)


def test_malformed_row_does_not_silently_shift_values():
    payload = raw_frame(1).to_csv(index=False) + "extra,private,values\n"
    with pytest.raises(DataValidationError, match="columnas"):
        load_csv(payload.encode())


def test_duplicate_header_is_rejected():
    raw = raw_frame(1)
    raw["currency_again"] = "USD"
    payload = raw.to_csv(index=False).replace("currency_again", "currency")
    with pytest.raises(DataValidationError, match="duplicadas"):
        load_csv(payload.encode())


def test_limits_are_checked(monkeypatch):
    payload = raw_frame().to_csv(index=False).encode()
    monkeypatch.setattr(data, "MAX_BYTES", 10)
    with pytest.raises(DataValidationError, match="200 MiB"):
        load_csv(payload)
    monkeypatch.setattr(data, "MAX_BYTES", len(payload) + 1)
    monkeypatch.setattr(data, "MAX_ROWS", 2)
    with pytest.raises(DataValidationError, match="200.000"):
        load_csv(payload)


def test_missing_columns_and_invalid_split_parameters():
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        prepare_frame(pd.DataFrame({"price": [1]}))
    frame = synthetic_data()
    for fraction in (0, 1, -1, np.nan):
        with pytest.raises(DataValidationError):
            split_data(frame, test_fraction=fraction)
    one_cell = frame.iloc[[0]].copy()
    with pytest.raises(DataValidationError, match="dos celdas"):
        split_data(one_cell)


def test_nonfinite_derived_values_are_rejected():
    raw = raw_frame(1)
    raw["surface_total_in_m2"] = 1e-300
    raw["surface_covered_in_m2"] = 1e-301
    raw["price"] = 1e300
    frame, report = prepare_frame(raw)
    assert frame.empty
    assert report["rejection_counts"]["invalid_derived_values"] == 1


def test_external_duplicate_indexes_do_not_misalign():
    raw = raw_frame()
    raw.index = [0, 0, 0]
    frame, _ = prepare_frame(raw)
    assert len(frame) == 3


@pytest.mark.parametrize("label", ["Palermo Hollywood", "Palermo Soho", "Palermo Chico", "Palermo Viejo"])
def test_known_subarea_labels_retained_without_geocoding(label):
    raw = raw_frame(1)
    raw["place_name"] = label
    frame, _ = prepare_frame(raw)
    assert frame.iloc[0]["neighborhood"] == label
    assert frame.iloc[0]["lat"] == raw.iloc[0]["lat"]


def test_large_ignored_free_text_field_is_valid_but_never_retained():
    raw = raw_frame(1)
    raw["description"] = "private long description, with quotes \"" * 4000
    frame, _ = load_csv(raw.to_csv(index=False).encode())
    assert len(frame) == 1
    assert "description" not in frame.columns


def test_split_rejects_invalid_recalculated_derived_values():
    frame = synthetic_data(n=20)
    frame.loc[0, ["price_usd", "area_total", "area_covered"]] = [1e300, 1e-300, 1e-301]
    with pytest.raises(DataValidationError, match="derivados"):
        split_data(frame)


def public_frame():
    canonical, _ = prepare_frame(raw_frame())
    return canonical.loc[:, data.PUBLIC_DEMO_COLUMNS].copy()


def test_load_demo_validates_and_returns_canonical_frame():
    public = public_frame()
    frame, report = load_demo(io.StringIO(public.to_csv(index=False)))
    assert tuple(frame.columns) == data.CANONICAL_COLUMNS
    assert len(frame) == 3
    assert frame.price_usd.tolist() == public.price_usd.tolist()
    assert report["coordinate_decimals"] == 3
    assert report["input_rows"] == report["accepted_rows"] == 3
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", ["extra", "missing", "reordered", "duplicate"])
def test_demo_requires_exact_public_header(mutation):
    public = public_frame()
    if mutation == "extra":
        public["address"] = "PRIVATE ADDRESS"
    elif mutation == "missing":
        public = public.drop(columns="rooms")
    elif mutation == "reordered":
        public = public.loc[:, list(reversed(public.columns))]
    else:
        public.columns = [*public.columns[:-1], "rooms"]
    with pytest.raises(DataValidationError, match="ocho columnas"):
        load_demo(io.StringIO(public.to_csv(index=False)))


def test_demo_rejects_precise_coordinates_and_filters_unknown_labels():
    public = public_frame()
    public.loc[0, "lat"] = -34.58041
    with pytest.raises(DataValidationError, match="tres decimales"):
        load_demo(io.StringIO(public.to_csv(index=False)))
    public["lat"] = public["lat"].round(3)
    public.loc[0, "neighborhood"] = "PRIVATE PERSON AND ADDRESS"
    frame, report = load_demo(io.StringIO(public.to_csv(index=False)))
    assert len(frame) == 2
    assert report["rejection_counts"]["unknown_neighborhood"] == 1
    assert "PRIVATE" not in frame.to_csv(index=False)


def test_demo_default_path_is_relative_to_module(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "demo.csv").write_text(public_frame().to_csv(index=False), encoding="utf-8")
    monkeypatch.setattr(data, "__file__", str(tmp_path / "data.py"))
    frame, report = load_demo()
    assert len(frame) == report["accepted_rows"] == 3


def test_preparation_is_reproducible_minimized_and_preserves_source(tmp_path):
    from scripts.prepare_demo import prepare_demo

    raw = raw_frame()
    raw.loc[0, "lat"] = -34.58041
    raw.loc[1, "lat"] = -34.58032
    raw.loc[1, "price"] = 110_000
    source, output = tmp_path / "source.csv", tmp_path / "demo.csv"
    source.write_text(raw.to_csv(index=False), encoding="utf-8")
    source_before = source.read_bytes()
    report = prepare_demo(source, output)
    first_output = output.read_bytes()
    assert source.read_bytes() == source_before
    assert report["source_valid_rows"] == 3
    assert report["rounding_duplicate_rows_removed"] == 1
    assert report["public_rows"] == 2
    assert report["sha256"] == hashlib.sha256(first_output).hexdigest()
    assert b"PRIVATE" not in first_output
    published = pd.read_csv(output, dtype=str)
    assert tuple(published.columns) == data.PUBLIC_DEMO_COLUMNS
    assert published["lat"].str.match(r"^-?\d+\.\d{3}$").all()
    assert published["lon"].str.match(r"^-?\d+\.\d{3}$").all()
    loaded, _ = load_demo(output)
    assert loaded.price_usd.tolist() == [100_000, 100_000]
    prepare_demo(source, output)
    assert output.read_bytes() == first_output
    json.dumps(report, allow_nan=False)


def test_preparation_never_overwrites_original(tmp_path):
    from scripts.prepare_demo import prepare_demo

    source = tmp_path / "source.csv"
    source.write_text(raw_frame().to_csv(index=False), encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(DataValidationError, match="original"):
        prepare_demo(source, source)
    assert source.read_bytes() == original


def test_preparation_rejects_hard_link_to_original(tmp_path):
    from scripts.prepare_demo import prepare_demo

    source, alias = tmp_path / "source.csv", tmp_path / "alias.csv"
    source.write_text(raw_frame().to_csv(index=False), encoding="utf-8")
    original = source.read_bytes()
    try:
        os.link(source, alias)
    except (PermissionError, NotImplementedError):
        pytest.skip("This execution environment does not permit creating hard links.")
    with pytest.raises(DataValidationError, match="original"):
        prepare_demo(source, alias)
    assert source.read_bytes() == original
