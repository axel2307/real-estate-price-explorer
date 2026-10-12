# Deployment

The public application is available at [real-estate-price-explorer.streamlit.app](https://real-estate-price-explorer.streamlit.app/), with source in [axel2307/real-estate-price-explorer](https://github.com/axel2307/real-estate-price-explorer). It uses a separate repository and deployment; the earlier geographic-comparables and linear-baseline projects remain intact.

## Review locally first

Use Python 3.12 and run from this project directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m compileall -q app.py explorer.py data.py comparables.py linear.py evaluation.py map_view.py scripts tests
python -m pytest
python -m streamlit run app.py
```

Review the unified form, two independent estimates, unsupported cases, shared evaluation, full-data map, point detail, pending edits and desktop/mobile layout. Submit zero and negative numeric inputs to confirm server validation clears old results. Verify that map filters preserve the color reference limits, the camera persists until an explicit recenter action, and endpoint colors leave record values accessible. Inspect the exact files to be published and confirm the current 7,495-row data fingerprint in [DATA_PROVENANCE.md](DATA_PROVENANCE.md), including the single authorized target-definition exclusion. Only the reviewed `data/demo.csv` belongs in the release; original inputs, private reports and credentials do not.

Review the local result, tests, evaluation and design before publishing an update. Record the reviewed commit and evidence separately from the runtime app. A display-only change must not silently replace or filter the dataset or reuse unverified metrics.

## Update or reproduce the deployment

1. For this project, use the existing `axel2307/real-estate-price-explorer` repository and its separate app. For another installation, create a repository you control; do not overwrite or repoint the two earlier projects.
2. Push the reviewed change and verify the **Tests** workflow on both Linux and Windows for that exact commit. A previous green run or a local run alone does not confirm that the update passed CI.
3. The Streamlit Community Cloud configuration is branch `main`, entry point `app.py`, and Python **3.12**. Use these settings when creating another installation. No secrets, Mapbox token, property-data account or upload environment flag are required. [Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
4. Verify the actual resulting URL in a fresh unauthenticated browser. Test desktop and mobile, initial data rendering, a valid comparison, each method's unsupported state, invalid areas, map filters and selection, and common evaluation. Confirm that displayed values match the reviewed local result.
5. Confirm that the public app is serving the reviewed update before announcing it. An existing live URL or successful build does not by itself verify that the latest behavior is available.

The app reads the bundled demonstration projection on startup. There is no upload mode. A missing or invalid projection is a deployment error and must not be hidden by substituting generated records. Carto basemap requests require browser network access; predictions do not depend on a property-data API.

## Reversibility

Keep the two existing applications available during the unified app's review and rollout. Use a separate deployment and retain the last verified commit. If a later change regresses behavior, restore the previously verified commit in this new repository and verify the redeployment. Do not delete or replace the original projects as part of that rollback. Data or methodology changes require their own review and a freshly computed evaluation.
