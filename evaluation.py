"""Compare unchanged models on a complete, shared spatial holdout.

Coverage uses every supplied held-out row. Errors for all three methods use
exactly the intersection supported by both predictors, with no random sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from comparables import ComparableModel, Filters, Query, _validate_frame
from linear import LinearPriceModel, NUMERICAL_LIMIT, fit_model


def _groups(frame: pd.DataFrame) -> frozenset[str]:
    if "spatial_group" not in frame:
        raise ValueError("La evaluación requiere grupos espaciales de entrenamiento y reserva.")
    values = frame["spatial_group"]
    if values.isna().any() or not values.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Los grupos espaciales deben ser identificadores válidos.")
    return frozenset(values)


def evaluate_joint(train: pd.DataFrame, test: pd.DataFrame,
                   filters: Filters | None = None,
                   fitted: LinearPriceModel | None = None) -> dict:
    """Evaluate all held-out rows; keep model coverage separate from errors.

    Inputs must come from the same validated spatial split. An optional fitted
    model must have been fitted on this exact training frame; its recorded group
    membership and row count are checked before use. Neither targets nor map
    filters are used to choose the evaluation population or model neighbours.
    The returned predictions retain the held-out frame's index and row order.
    """
    train_groups, test_groups = _groups(train), _groups(test)
    if train_groups & test_groups:
        raise ValueError("Entrenamiento y evaluación comparten grupos espaciales.")
    if "row_id" in train and "row_id" in test:
        if set(train["row_id"]) & set(test["row_id"]):
            raise ValueError("Entrenamiento y evaluación comparten identificadores de fila.")
    comparable_model = ComparableModel(train, filters)
    training_targets = train["usd_m2"].to_numpy(dtype=float)
    if (training_targets > NUMERICAL_LIMIT).any():
        raise ValueError("Los precios de entrenamiento superan el límite numérico del modelo lineal.")
    if not test.empty:
        _validate_frame(test)
    if "usd_m2" not in test:
        raise ValueError("La evaluación requiere precios observados en USD/m².")
    observed = pd.to_numeric(test["usd_m2"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(observed) & (observed > 0) & (observed <= NUMERICAL_LIMIT)).all():
        raise ValueError("Los precios observados deben ser finitos y positivos.")
    model = fit_model(train) if fitted is None else fitted
    if model.training_groups != train_groups or model.training_rows != len(train):
        raise ValueError("El modelo lineal no corresponde a los grupos y filas de entrenamiento.")
    linear_predictions = model.predict(test)
    comparable_predictions = [
        comparable_model.predict(Query(
            row.property_type, row.area_total, row.area_total * row.ratio,
            row.rooms, row.lat, row.lon,
        ))
        for row in test.itertuples(index=False)
    ]
    comparable_supported = np.array([value.sufficient for value in comparable_predictions], dtype=bool)
    linear_supported = linear_predictions["supported"].to_numpy(dtype=bool)
    common = comparable_supported & linear_supported
    comparable_values = np.array([
        value.usd_m2 if value.sufficient else np.nan for value in comparable_predictions
    ], dtype=float)
    linear_values = linear_predictions["predicted_usd_m2"].to_numpy(dtype=float)
    baseline = float(np.median(training_targets))
    predictions = pd.DataFrame({
        "observed_usd_m2": observed,
        "comparable_usd_m2": comparable_values,
        "linear_usd_m2": linear_values,
        "comparable_supported": comparable_supported,
        "linear_supported": linear_supported,
        "common_supported": common,
        "linear_reason": linear_predictions["reason"].to_numpy(),
    }, index=test.index)
    metrics = []
    for method, values in (
        ("comparables", comparable_values),
        ("linear", linear_values),
        ("training_median", np.full(len(test), baseline)),
    ):
        errors = np.abs(values[common] - observed[common])
        metrics.append({
            "method": method,
            "mae": float(errors.mean()) if len(errors) else None,
            "median_ae": float(np.median(errors)) if len(errors) else None,
        })
    total = len(test)
    return {
        "total_rows": total,
        "comparable_supported_rows": int(comparable_supported.sum()),
        "linear_supported_rows": int(linear_supported.sum()),
        "common_rows": int(common.sum()),
        "comparable_coverage": float(comparable_supported.mean()) if total else 0.0,
        "linear_coverage": float(linear_supported.mean()) if total else 0.0,
        "common_coverage": float(common.mean()) if total else 0.0,
        "baseline_train_median_usd_m2": baseline,
        "metrics": metrics,
        "predictions": predictions,
    }
