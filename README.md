# SPG Explorer — app de Streamlit

Analiza cualquier red compleja con el marco SPG: persistencia espectral,
comparación con centralidades clásicas, y el hallazgo central del proyecto
(dependencia del operador de difusión).

Basado en: Arteaga Marroquín, E. A. (2026). *A Systematic Empirical
Evaluation of Spectral Persistence Observables in Complex Networks*.
Zenodo. https://doi.org/10.5281/zenodo.21815650

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube este archivo (`app.py`) y `requirements.txt` a un repositorio de GitHub.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. "New app" → selecciona el repo → archivo principal: `app.py` → Deploy.

La app queda pública en una URL tipo `https://tu-app.streamlit.app`.
