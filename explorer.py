"""Shared orchestration; the original two statistical methods stay independent."""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from comparables import ComparableModel, Filters, Query
from data import LAT_BOUNDS, LON_BOUNDS, load_demo, split_data
from linear import fit_model


PROPERTY_LABELS = {"apartment": "Departamento", "house": "Casa", "PH": "PH"}
REASONS = {
    "invalid_features": "Revisá las superficies, los ambientes y la proporción cubierta.",
    "unknown_property_type": "El tipo de propiedad no tiene ejemplos de entrenamiento.",
    "unknown_neighborhood": "El barrio no tiene ejemplos de entrenamiento.",
    "unseen_category_combination": "No hay ejemplos de ese tipo de propiedad en ese barrio.",
    "outside_training_range": "Los valores están fuera de los rangos de entrenamiento.",
    "nonpositive_or_nonfinite_prediction": "El modelo no produjo un precio positivo y finito.",
}


@dataclass(frozen=True)
class PropertyQuery:
    property_type: str
    neighborhood: str
    rooms: int
    area_total: float
    area_covered: float
    lat: float
    lon: float

    def validate(self):
        query = Query(self.property_type, self.area_total, self.area_covered,
                      self.rooms, self.lat, self.lon)
        if self.property_type not in PROPERTY_LABELS:
            raise ValueError("Elegí un tipo de propiedad disponible.")
        if not isinstance(self.neighborhood, str) or not self.neighborhood.strip():
            raise ValueError("Elegí un barrio.")
        if not LAT_BOUNDS[0] <= self.lat <= LAT_BOUNDS[1] or not LON_BOUNDS[0] <= self.lon <= LON_BOUNDS[1]:
            raise ValueError("La ubicación debe estar dentro del área disponible de CABA.")
        return query

    def as_frame(self):
        self.validate()
        return pd.DataFrame([{**asdict(self), "ratio": self.area_covered / self.area_total}])


def prepare_demo():
    """Assign stable display IDs after the unchanged spatial split.

    IDs are positions in the public train/test union, never original source IDs.
    Keeping each partition's order preserves stable distance tie-breaking.
    """
    validated, report = load_demo()
    train, test, split_info = split_data(validated, test_fraction=0.2, seed=42)
    frame = pd.concat([train, test], ignore_index=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    train = frame.iloc[:len(train)].copy()
    test = frame.iloc[len(train):].copy()
    fitted = fit_model(train)
    return frame, train, test, fitted, report, split_info


def neighborhood_location(train, neighborhood):
    """An explicit reference from training examples, not an official centroid."""
    examples = train.loc[train["neighborhood"].eq(neighborhood)]
    if examples.empty:
        raise ValueError("Este barrio no tiene ubicaciones de entrenamiento.")
    return round(float(examples["lat"].median()), 3), round(float(examples["lon"].median()), 3)


def compare_property(train, fitted, query: PropertyQuery, filters: Filters):
    geographic_query = query.validate()
    comparable = ComparableModel(train, filters).predict(geographic_query)
    linear = fitted.predict(query.as_frame()).iloc[0].to_dict()
    difference = None
    if comparable.sufficient and linear["supported"]:
        delta = comparable.usd_m2 - float(linear["predicted_usd_m2"])
        total_delta = comparable.asking_price_usd - float(linear["predicted_price_usd"])
        if isfinite(delta) and isfinite(total_delta):
            difference = {"usd_m2": delta, "total_usd": total_delta}
    return {"query": query, "filters": filters, "comparables": comparable,
            "linear": linear, "difference": difference}


def money(value, decimals=0):
    value = f"{value:,.{decimals}f}"
    return "USD " + value.replace(",", "_").replace(".", ",").replace("_", ".")


def display_rows(frame):
    """Only explicitly authorized public fields reach a record table."""
    columns = ["row_id", "property_type", "neighborhood", "rooms", "area_total",
               "area_covered", "price_usd", "usd_m2"]
    view = frame.loc[:, columns].copy()
    view["property_type"] = view["property_type"].map(PROPERTY_LABELS)
    return view.rename(columns={
        "row_id": "Registro", "property_type": "Tipo", "neighborhood": "Barrio",
        "rooms": "Ambientes", "area_total": "Total (m²)", "area_covered": "Cubierta (m²)",
        "price_usd": "Precio (USD)", "usd_m2": "USD/m²",
    })
