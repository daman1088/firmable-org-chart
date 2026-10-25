# Deploy the Firmable Org Chart on Streamlit Community Cloud

This dashboard loads the precomputed 600-company synthetic dataset and fitted model from the public GitHub Release `demo-data-v1`. It does **not** regenerate or train the model in Streamlit Cloud.

1. Go to https://share.streamlit.io and deploy the GitHub repo `daman1088/firmable-org-chart`, branch `main`, main file `app/streamlit_app.py`.
2. In Advanced settings select Python **3.12** when possible.
3. Under app Secrets add:

   ```toml
   ORGCHART_AUTOBUILD = "1"
   ```

   This is a non-secret flag that enables downloading the release if the data is absent.
4. Deploy or reboot. The first startup downloads `https://github.com/daman1088/firmable-org-chart/releases/download/demo-data-v1/demo-data.zip` and extracts `data/full`; later reruns reuse the files on that instance.
5. Check company selection, org chart, Ask the Org Chart, Employees and Precision targets. Copy the deployed app URL into the submission.

The repo intentionally excludes the `data/` directory; its generator recreates this synthetic dataset. The public release asset also includes synthetic ground truth used for evaluation. Do not put real people, private customer records, or secrets in the release.

If the cloud service resets its ephemeral disk, the next app launch downloads the release again. For local development, use `make setup`, `make pipeline GENERATE=1`, `make app`.
