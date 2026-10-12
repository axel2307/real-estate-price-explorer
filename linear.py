"""Small, auditable OLS baseline for dataset price targets in USD/m².

All transformations are learned from the training frame. Prediction accepts named
columns and returns an explicit support decision for each row. No intervals are
inferred from training residuals. The bundled data's fictional status is the
author's declaration; this model makes no claim about market prices.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_FEATURES = ("rooms", "area_total", "ratio")
CATEGORICAL_FEATURES = ("property_type", "neighborhood")
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "usd_m2"
# A numerical safety bound, deliberately far beyond plausible market values.
# It keeps reductions, squared deviations and derived totals away from overflow.
NUMERICAL_LIMIT = 1e100


class ModelValidationError(ValueError):
    """The supplied frame cannot safely be used to fit or query this model."""


def _check_columns(frame: pd.DataFrame, target: bool = False) -> None:
    expected = REQUIRED_FEATURES + ((TARGET,) if target else ())
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ModelValidationError("Missing required columns: " + ", ".join(missing))
    if not frame.columns.is_unique:
        raise ModelValidationError("Column names must be unique.")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce")


def _invalid_numeric(values: pd.DataFrame) -> np.ndarray:
    return (
        ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        | (np.abs(values.to_numpy(dtype=float)) > NUMERICAL_LIMIT).any(axis=1)
        | (values["area_total"].to_numpy() <= 0)
        | (values["rooms"].to_numpy() < 1)
        | (values["rooms"].to_numpy() != np.floor(values["rooms"].to_numpy()))
        | (values["ratio"].to_numpy() <= 0)
        | (values["ratio"].to_numpy() > 1)
    )


@dataclass(frozen=True)
class LinearPriceModel:
    """Fitted, named OLS design, including its learned support boundaries."""

    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    means: pd.Series
    scales: pd.Series
    lower: pd.Series
    upper: pd.Series
    category_pairs: frozenset[tuple[str, str]]
    training_groups: frozenset[str]
    rank: int
    condition_number: float
    training_rows: int
    baseline_usd_m2: float

    @property
    def rank_deficient(self) -> bool:
        return self.rank < len(self.feature_names)

    @property
    def reference_categories(self) -> dict[str, str]:
        return {column: levels[0] for column, levels in self.categories.items()}

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = _numeric_frame(frame)
        columns: dict[str, np.ndarray] = {"intercept": np.ones(len(frame))}
        for column in NUMERIC_FEATURES:
            columns["numeric:" + column] = (
                (numeric[column] - self.means[column]) / self.scales[column]
            ).to_numpy(dtype=float)
        for column in CATEGORICAL_FEATURES:
            for level in self.categories[column][1:]:
                columns[f"category:{column}={level}"] = (
                    frame[column].astype(str).eq(level).to_numpy(dtype=float)
                )
        # This ordering comes only from the fitted model, never the input frame.
        return np.column_stack([columns[name] for name in self.feature_names])

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return predictions only inside the learned feature support.

        Reasons have a deterministic precedence: invalid features, unknown
        categories, unseen category pair, numeric extrapolation, then nonpositive
        model output. This reports unsupported data instead of silently clipping.
        """
        _check_columns(frame)
        numeric = _numeric_frame(frame)
        count = len(frame)
        reason = np.full(count, "", dtype=object)
        invalid = _invalid_numeric(numeric)
        if "area_covered" in frame.columns:
            covered = pd.to_numeric(frame["area_covered"], errors="coerce").to_numpy(dtype=float)
            total = numeric["area_total"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = covered / total
            invalid |= (
                ~np.isfinite(covered) | (covered <= 0) | (covered > total)
                | ~np.isclose(ratio, numeric["ratio"].to_numpy(), rtol=1e-6, atol=1e-8)
            )
        reason[invalid] = "invalid_features"
        for column in CATEGORICAL_FEATURES:
            unknown = ~frame[column].isin(self.categories[column]).to_numpy()
            reason[(reason == "") & unknown] = "unknown_" + column
        pairs = zip(frame["property_type"].astype(str), frame["neighborhood"].astype(str))
        unknown_pair = np.array([pair not in self.category_pairs for pair in pairs], dtype=bool)
        reason[(reason == "") & unknown_pair] = "unseen_category_combination"
        tolerance = 1e-9 * np.maximum(1.0, np.abs(self.upper.to_numpy(dtype=float)))
        outside = (
            (numeric.to_numpy(dtype=float) < self.lower.to_numpy(dtype=float) - tolerance)
            | (numeric.to_numpy(dtype=float) > self.upper.to_numpy(dtype=float) + tolerance)
        ).any(axis=1)
        reason[(reason == "") & outside] = "outside_training_range"
        predicted = np.full(count, np.nan)
        candidates = np.flatnonzero(reason == "")
        if len(candidates):
            with np.errstate(over="ignore", invalid="ignore"):
                values = self._design(frame.iloc[candidates]) @ self.coefficients
                implied_total = values * numeric["area_total"].to_numpy(dtype=float)[candidates]
            bad = (
                ~np.isfinite(values) | (values <= 0)
                | ~np.isfinite(implied_total) | (implied_total <= 0)
            )
            reason[candidates[bad]] = "nonpositive_or_nonfinite_prediction"
            predicted[candidates[~bad]] = values[~bad]
        return pd.DataFrame(
            {
                "predicted_usd_m2": predicted,
                "predicted_price_usd": predicted * numeric["area_total"].to_numpy(dtype=float),
                "supported": reason == "",
                "reason": reason,
            },
            index=frame.index,
        )

    def evaluate(self, heldout: pd.DataFrame) -> dict[str, Any]:
        """Evaluate model and training median on exactly the same held-out rows."""
        _check_columns(heldout, target=True)
        if self.training_groups:
            if "spatial_group" not in heldout.columns:
                raise ModelValidationError("Held-out spatial groups are required for evaluation.")
            evaluation_groups = frozenset(heldout["spatial_group"].astype(str))
            if self.training_groups & evaluation_groups:
                raise ModelValidationError("Training and held-out spatial groups overlap.")
        prediction = self.predict(heldout)
        observed = pd.to_numeric(heldout[TARGET], errors="coerce").to_numpy(dtype=float)
        valid_target = np.isfinite(observed) & (observed > 0) & (observed <= NUMERICAL_LIMIT)
        supported = prediction["supported"].to_numpy() & valid_target
        reasons = prediction["reason"].to_numpy(copy=True)
        reasons[~valid_target] = "invalid_target"
        total = len(heldout)
        result: dict[str, Any] = {
            "total_rows": total,
            "supported_rows": int(supported.sum()),
            "coverage": float(supported.mean()) if total else 0.0,
            "mae_usd_m2": None,
            "median_ae_usd_m2": None,
            "baseline_mae_usd_m2": None,
            "baseline_median_ae_usd_m2": None,
            "excluded_reasons": pd.Series(reasons[~supported], dtype=str).value_counts().to_dict(),
        }
        if supported.any():
            errors = np.abs(prediction["predicted_usd_m2"].to_numpy()[supported] - observed[supported])
            baseline_errors = np.abs(self.baseline_usd_m2 - observed[supported])
            result.update(
                mae_usd_m2=float(errors.mean()),
                median_ae_usd_m2=float(np.median(errors)),
                baseline_mae_usd_m2=float(baseline_errors.mean()),
                baseline_median_ae_usd_m2=float(np.median(baseline_errors)),
            )
        return result


def fit_model(training: pd.DataFrame) -> LinearPriceModel:
    """Fit scaled OLS with one intercept and one reference per category."""
    _check_columns(training, target=True)
    if len(training) < 20:
        raise ModelValidationError("At least 20 valid training rows are required.")
    numeric = _numeric_frame(training)
    target = pd.to_numeric(training[TARGET], errors="coerce").to_numpy(dtype=float)
    if _invalid_numeric(numeric).any() or not (np.isfinite(target) & (target > 0) & (target <= NUMERICAL_LIMIT)).all():
        raise ModelValidationError("Training rows need finite positive targets and valid features within the numerical safety limit (1e100).")
    categories: dict[str, tuple[str, ...]] = {}
    names = ["intercept"] + ["numeric:" + column for column in NUMERIC_FEATURES]
    for column in CATEGORICAL_FEATURES:
        if training[column].isna().any() or not training[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ModelValidationError("Training categories must be nonempty strings.")
        categories[column] = tuple(sorted(training[column].unique()))
        names.extend(f"category:{column}={level}" for level in categories[column][1:])
    if len(training) <= len(names):
        raise ModelValidationError("More training rows than model columns are required.")
    means = numeric.mean()
    scales = numeric.std(ddof=0).replace(0.0, 1.0)
    common = dict(
        feature_names=tuple(names), categories=categories, means=means, scales=scales,
        lower=numeric.min(), upper=numeric.max(),
        category_pairs=frozenset(zip(training["property_type"], training["neighborhood"])),
        training_groups=frozenset(training["spatial_group"].astype(str)) if "spatial_group" in training else frozenset(),
        training_rows=len(training), baseline_usd_m2=float(np.median(target)),
    )
    template = LinearPriceModel(coefficients=np.zeros(len(names)), rank=0, condition_number=float("inf"), **common)
    design = template._design(training)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if not np.isfinite(coefficients).all():
        raise ModelValidationError("The numerical fit did not produce finite coefficients.")
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == len(names) and len(singular_values) and singular_values[-1] > 0
        else float("inf")
    )
    return LinearPriceModel(coefficients=coefficients, rank=int(rank), condition_number=condition, **common)
