# 🕸️ SPG Explorer

An interactive tool to analyze the **spectral persistence** of any complex
network — how well each node retains a perturbation under diffusion, how
that compares to classical centrality measures, and how it depends on the
diffusion operator chosen.

This app is a companion to the research project
**["A Systematic Empirical Evaluation of Spectral Persistence Observables
in Complex Networks"](https://github.com/EdherAlanArteaga/spectral-persistence-audit)**
(Arteaga Marroquín, 2026 — [DOI 10.5281/zenodo.21815650](https://doi.org/10.5281/zenodo.21815650)).
That project audited a set of conjectures about this observable; most were
refuted, and the few that survived — including the operator-dependence
result this app highlights — are reported honestly alongside what didn't
hold up. Read the paper for the full context before drawing conclusions
from this tool.

## What you can do with it

Upload any network as a simple edge-list CSV (or try a built-in sample
network) and get:

- **Degree vs. persistence** — a log-log plot, plus how much of the effect
  is simple 1/degree scaling (with R²).
- **Node table** — every node ranked by persistence, top/bottom 10
  highlighted, downloadable as CSV.
- **Comparison against classical centralities** — degree, eigenvector,
  closeness, current-flow closeness, betweenness — with rank correlations.
- **Operator dependence** — the project's central finding, made
  interactive: a slider shows how the degree–persistence anti-correlation
  collapses as the diffusion operator is normalized (γ = 0 → 0.5).
- **Temporal dynamics** — how well persistence predicts a node's
  self-retention at different diffusion timescales.

No data is stored or sent anywhere — everything runs in your browser
session (or your own machine, if run locally).

## Try it

*(Add your deployed Streamlit Cloud URL here once live, e.g.
`https://spg-explorer.streamlit.app`)*

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy your own copy (Streamlit Community Cloud)

1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. "New app" → select the repo → main file: `app.py` → Deploy.

## Related

- 📄 [Full paper (Zenodo, DOI)](https://doi.org/10.5281/zenodo.21815650)
- 💻 [Research code and notebooks (GitHub)](https://github.com/EdherAlanArteaga/spectral-persistence-audit)

## Author

Edher Alan Arteaga Marroquín — [ORCID 0009-0004-7333-1975](https://orcid.org/0009-0004-7333-1975)

## License

MIT
