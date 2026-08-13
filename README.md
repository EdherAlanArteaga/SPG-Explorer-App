# SPG Explorer — Streamlit app

Analyze any complex network with the SPG framework: spectral persistence,
comparison against classical centralities, and the project's central
finding (dependence on the diffusion operator).

Based on: Arteaga Marroquín, E. A. (2026). *A Systematic Empirical
Evaluation of Spectral Persistence Observables in Complex Networks*.
Zenodo. https://doi.org/10.5281/zenodo.21815650

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this `app.py` and `requirements.txt` to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. "New app" → select the repo → main file: `app.py` → Deploy.

The app becomes public at a URL like `https://your-app.streamlit.app`.
