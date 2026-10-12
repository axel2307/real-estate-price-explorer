# Data and privacy

The application uses only the bundled projection in `data/demo.csv`. The project author supplied the source, declared it fictional, and explicitly authorized publication of this minimized projection. That declaration has not been independently verified. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) records the source statement and transformation.

## What is published

The eight fields are `property_type`, `neighborhood`, `area_total`, `area_covered`, `rooms`, `lat`, `lon`, and `price_usd`. Coordinates are rounded to three decimals before a second validation and duplicate check. This reduces precision; it does not prove anonymization. Unit price, covered ratio and spatial groups are derived at runtime.

The full source is not distributed. Addresses, names, contacts, descriptions, titles, URLs, images, source identifiers, unused fields and private audit materials are excluded. Text labels are restricted to known property types and allowlisted CABA neighborhoods/subareas. No external property-data provider or geocoder is called.

## What the browser receives

The explorer exposes all 7,496 published demonstration records through the map and record detail controls. Coincident coordinates are grouped visually, with access to the records at that point. Grouping does not remove or anonymize those rows. Public price and feature values can be viewed; do not replace the bundled projection with confidential data under the assumption that only aggregate values will be visible.

The query marker and selected neighbor locations are displayed at three-decimal precision. Query values are sent to the Streamlit server to calculate the two estimates and remain in the visitor's session state while the session is active. The application does not write queries to disk or deliberately log their contents. No street address, name or contact field is requested.

Shared caches contain the public demonstration data, its fitted model, evaluation results for comparable settings, and filtered/grouped public map records. Individual property queries and their estimates are kept in session state, not in those caches.

Pydeck loads the Carto basemap through the browser. That service can receive ordinary request metadata, including IP address and the viewed map region. The app does not send a private property dataset to a property-data provider. This configuration needs no map token. Browser and hosting infrastructure have their own storage and logging behavior; application-level behavior is not a retention guarantee for those services.

## Uploads and local preparation

This version has no upload interface and no setting to enable uploads. The raw CSV/TSV parser remains a local preparation and testing utility. Running `scripts/prepare_demo.py` requires an explicit local source path; it is not part of a visitor request and refuses to overwrite its source. Review rights and privacy before preparing or publishing any replacement dataset.

The only data-file exception in `.gitignore` is `data/demo.csv`. Before release, inspect the exact staged file list: original inputs, local paths in private reports, credentials, transfer metadata and browser artifacts must remain outside the repository.
