# Hosted app deployment (Streamlit Community Cloud)

The hosted app uses the same 600-company generator, fitted model and inference pipeline as the local demonstration.

## Deploy

1. Open https://share.streamlit.io/ and sign in with GitHub. Authorize access to this private repository if prompted.
2. Choose **Create app / Deploy from GitHub**, repository `daman1088/firmable-org-chart`, branch `main`, and main file path `app/streamlit_app.py`.
3. Under **Advanced settings**, select **Python 3.12** and paste the following into **Secrets**:

   ```toml
   ORGCHART_AUTOBUILD = "1"
   ```

   This is a configuration flag, not a sensitive credential. Root-level Streamlit secrets are exposed as environment variables.
4. Set a suitable app URL, deploy, and leave the initial build running until it finishes.
5. Test company selection, the Org chart tab, Ask the Org Chart, Employees and Precision targets. Copy the live URL into your submission.

## Important behavior

* Generated data is intentionally not committed to Git. On a new cloud instance, `ORGCHART_AUTOBUILD = "1"` triggers `orgchart.cli.pipeline(... generate_data=True)` on first launch to create the dataset, normalization files, models, predictions and evaluation reports inside `data/full`.
* First startup may be slow; subsequent reruns on the same instance reuse files. A cloud restart or redeploy that deletes the instance's ephemeral disk will rebuild them.
* Building the entire pipeline on Community Cloud is **best-effort**: constrained hosted instances can hit memory or execution limits. Check the Cloud deployment logs if generation fails. For a production/reviewer-critical hosted build, a persistent artifact store or a host with enough memory and disk is more robust.
* The local workflow is unchanged: `make setup`, `make pipeline GENERATE=1`, `make app`.
* The publicly accessible dashboard uses synthetic data; never upload credentials or real employee data.
* To let Firmable review the Git source separately, invite them to the private GitHub repository, or make the repository public if the assessment terms allow.
