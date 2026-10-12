# Deployment

Deploy this project as a separate Streamlit application. Its intended repository name is `real-estate-price-explorer`; verify availability before creating it. Preserve the existing geographic-comparables and linear-baseline repositories and their deployments. This document describes deployment steps, not evidence that a public app already exists.

## Review locally first

Use Python 3.12 and run from this project directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m compileall -q app.py explorer.py data.py comparables.py linear.py evaluation.py map_view.py scripts tests
python -m pytest
python -m streamlit run app.py
```

Review the unified form, two independent estimates, unsupported cases, shared evaluation, full-data map, point detail, pending edits and desktop/mobile layout. Inspect the exact files to be published and confirm the data fingerprint in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Only the reviewed `data/demo.csv` belongs in the release; original inputs, private reports and credentials do not.

The initial release requires review of the local result, tests, evaluation and design before publication. Record the reviewed commit and evidence separately from the runtime app. Do not publish an unreviewed work-in-progress just to obtain a preview URL.

## Create the separate deployment

1. Create a new GitHub repository under the approved account using an available, agreed name. If the intended name already exists, inspect it and resolve the destination; do not overwrite it or repoint an existing app.
2. Push the reviewed project and verify the **Tests** workflow on both Linux and Windows. A local run alone does not confirm that CI passed.
3. In Streamlit Community Cloud, create a new app with the new repository, branch `main`, entry point `app.py`, and Python **3.12** in Advanced settings. No secrets, Mapbox token, property-data account or upload environment flag are required. [Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
4. Verify the actual resulting URL in a fresh unauthenticated browser. Test desktop and mobile, initial data rendering, a valid comparison, each method's unsupported state, invalid areas, map filters and selection, and common evaluation. Confirm that displayed values match the reviewed local result.
5. Only after verification, add the real app/repository links to the README and share them. A chosen subdomain or successful build does not by itself verify the complete visitor experience.

The app reads the bundled demonstration projection on startup. There is no upload mode. A missing or invalid projection is a deployment error and must not be hidden by substituting generated records. Carto basemap requests require browser network access; predictions do not depend on a property-data API.

## Reversibility

Keep the two existing applications available during the unified app's review and rollout. Use a separate deployment and retain the last verified commit. If a later change regresses behavior, restore the previously verified commit in this new repository and verify the redeployment. Do not delete or replace the original projects as part of that rollback. Data or methodology changes require their own review and a freshly computed evaluation.
