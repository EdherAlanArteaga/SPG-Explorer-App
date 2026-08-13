"""
SPG Explorer — Análisis de persistencia espectral en redes complejas
======================================================================
Basado en: "A Systematic Empirical Evaluation of Spectral Persistence
Observables in Complex Networks" (Arteaga Marroquín, 2026)
DOI: 10.5281/zenodo.21815650

Para correr localmente:
    pip install streamlit numpy pandas networkx scipy matplotlib
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh, expm
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPG Explorer", layout="wide", page_icon="🕸️")

# ----------------------------------------------------------------------
# Motor de cálculo
# ----------------------------------------------------------------------

def cargar_grafo(df):
    df = df.iloc[:, :2]
    df.columns = ['s', 't']
    G = nx.Graph()
    G.add_edges_from(df.values)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def espectro(L):
    N = L.shape[0]
    ev, evec = eigh(L)
    k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
    return ev, evec, k0


def V_de_L(L):
    ev, evec, k0 = espectro(L)
    if k0 is None:
        return None, ev, evec, k0
    N = L.shape[0]
    V = np.zeros(N)
    for k in range(k0, N):
        V += evec[:, k] ** 2 / ev[k]
    return V, ev, evec, k0


def tau_tilde_de_L(V, ev, evec, k0):
    N = evec.shape[0]
    M2 = np.zeros(N)
    for k in range(k0, N):
        M2 += evec[:, k] ** 2 / ev[k] ** 2
    tau = np.where(V > 1e-14, M2 / V, 0)
    return ev[k0] * tau


def L_gamma(A, gamma):
    deg = A.sum(1)
    L = np.diag(deg) - A
    Dg = np.diag(deg ** (-gamma))
    return Dg @ L @ Dg


# ----------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------

st.title("🕸️ SPG Explorer")
st.markdown(
    "Analiza la **persistencia espectral** de cualquier red compleja: qué tan "
    "bien un nodo retiene una perturbación bajo difusión, cómo se compara con "
    "el grado y con centralidades clásicas, y cómo depende del operador de "
    "difusión elegido — el hallazgo central de este trabajo."
)

with st.expander("📖 ¿Qué mide esto, exactamente? — leer antes de interpretar resultados"):
    st.markdown("""
El observable central, **V_i = Σ v_k(i)²/λ_k**, es matemáticamente idéntico a
**L⁺_ii**, la diagonal de la pseudoinversa del Laplaciano — un objeto conocido
en teoría espectral de grafos (Van Mieghem, Devriendt & Cetinay, *Phys. Rev. E*
96, 032311, 2017), usado ahí como medida de capacidad de propagación.

**Verificado empíricamente sobre 30+ redes reales, biológicas y no biológicas:**
- En redes con grado heterogéneo, V anti-correlaciona fuerte con el grado.
- Esa anti-correlación **depende del operador de difusión**, no es fija de la
  red: colapsa al pasar del Laplaciano combinatorio al normalizado simétrico.
- El efecto se explica en gran parte por V≈c/grado, aunque el grado de ajuste
  varía sustancialmente entre redes (R² de 0.38 a 1.00 en el estudio original).
- τ̃ (un observable derivado) se probó redundante con V en tareas de
  estructura de comunidades.

**Esta app no afirma que V sea un observable nuevo**, ni que supere a las
centralidades clásicas en tareas prácticas (de hecho, en un benchmark de
colocación de sensores, no lo hizo — ver paper, Sección 8).

[Ver paper completo (Zenodo, DOI)](https://doi.org/10.5281/zenodo.21815650) · 
[Ver código y notebooks (GitHub)](https://github.com/EdherAlanArteaga/spectral-persistence-audit)
""")

st.divider()

# --- carga de datos ---
st.subheader("1️⃣ Sube tu red")
st.caption(
    "CSV de aristas: cada fila un par (nodo_origen, nodo_destino). "
    "Compatible directo con archivos `edges.csv` de Netzschleuder (networks.skewed.de)."
)

col_up, col_demo = st.columns([2, 1])
with col_up:
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
with col_demo:
    demo_choice = st.selectbox(
        "…o usa una red de ejemplo",
        ["(ninguna)", "Karate Club (34 nodos)", "Erdős–Rényi aleatoria (150 nodos)",
         "Barabási–Albert / scale-free (150 nodos)"]
    )

G = None
if archivo is not None:
    try:
        df = pd.read_csv(archivo, comment='#', header=None)
        G = cargar_grafo(df)
        st.success(f"✅ Red cargada: **{G.number_of_nodes()}** nodos, "
                   f"**{G.number_of_edges()}** aristas (componente conexa mayor).")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
elif demo_choice != "(ninguna)":
    if "Karate" in demo_choice:
        G = nx.karate_club_graph()
    elif "Erdős" in demo_choice:
        G = nx.erdos_renyi_graph(150, 0.06, seed=1)
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    elif "Barabási" in demo_choice:
        G = nx.barabasi_albert_graph(150, 2, seed=1)
    st.info(f"Usando red de ejemplo: {demo_choice} — {G.number_of_nodes()} nodos.")

if G is None:
    st.warning("Sube un CSV de aristas o elige una red de ejemplo para continuar.")
    st.stop()

N = G.number_of_nodes()
if N < 4:
    st.error("La red necesita al menos 4 nodos conectados.")
    st.stop()
grande = N > 800
if N > 3000:
    st.warning(f"La red tiene {N} nodos — el cálculo puede tardar varios minutos.")

nodes = list(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(1)
L0 = np.diag(deg) - A

with st.spinner("Calculando espectro y observables…"):
    V, ev, evec, k0 = V_de_L(L0)
    if V is None:
        st.error("La red no tiene un modo de Fiedler bien definido.")
        st.stop()
    tau_t = tau_tilde_de_L(V, ev, evec, k0)

sp_rho, _ = spearmanr(deg, V)
pe_rho, _ = pearsonr(V, 1 / deg)
r2 = pe_rho ** 2
density = A.sum() / (N * (N - 1))
cv_deg = deg.std() / deg.mean()

# ----------------------------------------------------------------------
# Panel resumen
# ----------------------------------------------------------------------
st.divider()
st.subheader("2️⃣ Resumen de la red")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodos", N)
c2.metric("Aristas", G.number_of_edges())
c3.metric("Densidad", f"{density:.4f}")
c4.metric("CV(grado)", f"{cv_deg:.2f}")
c5.metric("Spearman(k, V)", f"{sp_rho:+.3f}")

if abs(sp_rho) > 0.7:
    veredicto = "🔴 Anti-centralidad fuerte: los nodos de mayor grado tienden a retener menos."
elif abs(sp_rho) > 0.3:
    veredicto = "🟡 Anti-centralidad moderada."
else:
    veredicto = "⚪ Sin anti-centralidad clara — posiblemente red con grado homogéneo."
st.markdown(f"**Veredicto:** {veredicto}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Grado vs. V", "🏆 Tabla de nodos", "🔬 Vs. centralidades clásicas",
    "🧭 Dependencia del operador", "🌊 Dinámica temporal"
])

# --- TAB 1: grado vs V ---
with tab1:
    st.markdown("### Grado vs. persistencia espectral")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))
    ax1.scatter(deg, V, alpha=0.6, s=28, color="#1a1a1a")
    ax1.set_xlabel("grado"); ax1.set_ylabel("V (= L⁺_ii)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_title("Escala log-log")
    st.pyplot(fig1)

    st.markdown("### ¿Cuánto de esto es simple escalamiento 1/grado?")
    fig1b, ax1b = plt.subplots(figsize=(6.5, 4.8))
    ax1b.scatter(1/deg, V, alpha=0.6, s=28, color="#555555")
    ax1b.set_xlabel("1 / grado"); ax1b.set_ylabel("V")
    ax1b.set_title(f"R² = {r2:.3f}")
    st.pyplot(fig1b)
    st.caption(
        f"Si R² es cercano a 1.0, V se explica casi todo por el escalamiento "
        f"trivial V≈c/grado. En el estudio original, este valor varió de 0.38 "
        f"a 1.00 entre 11 redes reales, con densidad como el mejor predictor "
        f"parcial encontrado (correlación moderada, no una ley limpia)."
    )

# --- TAB 2: tabla de nodos ---
with tab2:
    st.markdown("### Nodos ordenados por persistencia (V)")
    tabla = pd.DataFrame({
        "nodo": nodes, "grado": deg.astype(int), "V": V, "τ̃": tau_t,
    }).sort_values("V", ascending=False).reset_index(drop=True)
    tabla.index += 1

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Top 10 — mayor persistencia (mejores 'retenedores')**")
        st.dataframe(tabla.head(10), use_container_width=True)
    with colB:
        st.markdown("**Bottom 10 — menor persistencia (mejores 'propagadores')**")
        st.dataframe(tabla.tail(10).sort_values("V"), use_container_width=True)

    st.markdown("### Tabla completa")
    st.dataframe(tabla, use_container_width=True, height=350)

    csv_out = tabla.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Descargar tabla completa (CSV)", data=csv_out,
                        file_name="spg_nodos.csv", mime="text/csv")

# --- TAB 3: vs centralidades clásicas ---
with tab3:
    st.markdown("### V frente a centralidades clásicas")
    if grande:
        st.warning(
            "Red grande (>800 nodos): betweenness y current-flow closeness "
            "pueden tardar bastante. Actívalo si tienes tiempo."
        )
        correr_completo = st.checkbox("Calcular de todos modos (puede tardar)")
    else:
        correr_completo = True

    if correr_completo:
        with st.spinner("Calculando centralidades clásicas…"):
            eig_c = nx.eigenvector_centrality_numpy(G)
            clo_c = nx.closeness_centrality(G)
            try:
                cfc_c = nx.current_flow_closeness_centrality(G)
            except Exception:
                cfc_c = {n: np.nan for n in nodes}
            btw_c = nx.betweenness_centrality(G) if N <= 2000 else None

        comp = pd.DataFrame({
            "grado": deg,
            "V": V,
            "eigenvector": [eig_c[n] for n in nodes],
            "closeness": [clo_c[n] for n in nodes],
            "current_flow_closeness": [cfc_c[n] for n in nodes],
        })
        if btw_c is not None:
            comp["betweenness"] = [btw_c[n] for n in nodes]

        st.markdown("**Correlación de rango (Spearman) de V contra cada métrica:**")
        filas = []
        for col in comp.columns:
            if col == "V":
                continue
            rho, _ = spearmanr(V, comp[col])
            filas.append({"métrica": col, "Spearman(V, ·)": round(rho, 3)})
        tabla_corr = pd.DataFrame(filas).sort_values("Spearman(V, ·)", key=abs, ascending=False)
        st.dataframe(tabla_corr, use_container_width=True, hide_index=True)

        cfc_rho = tabla_corr.loc[tabla_corr["métrica"] == "current_flow_closeness", "Spearman(V, ·)"]
        if len(cfc_rho) and abs(cfc_rho.values[0]) > 0.95:
            st.info(
                "V ordena los nodos casi idéntico a *current-flow closeness centrality* "
                "— están fuertemente relacionadas pero **no son la misma cantidad** "
                "(difieren en valor exacto, no solo en orden; ver paper Sección 4)."
            )
    else:
        st.info("Activa la casilla para calcular las centralidades clásicas.")

# --- TAB 4: dependencia del operador ---
with tab4:
    st.markdown("### ¿La anti-correlación es de la red, o del operador de difusión?")
    st.caption(
        "L_γ = D⁻ᵞ(D−A)D⁻ᵞ. γ=0 → Laplaciano combinatorio (diagonal = grado). "
        "γ=0.5 → Laplaciano normalizado simétrico (diagonal uniforme = 1). "
        "**Este es el hallazgo más sólido del proyecto original.**"
    )
    gamma_max = st.slider("Calcular hasta γ =", 0.1, 0.5, 0.5, 0.05, key="gamma_slider")
    n_pasos = st.slider("Resolución (nº de puntos)", 5, 25, 12)

    if st.button("▶️ Correr barrido de γ"):
        gammas = np.linspace(0, gamma_max, n_pasos)
        rhos = []
        barra = st.progress(0)
        for i, g in enumerate(gammas):
            Lg = L_gamma(A, g)
            Vg, _, _, k0g = V_de_L(Lg)
            rho_g = spearmanr(Vg, deg)[0] if Vg is not None else np.nan
            rhos.append(rho_g)
            barra.progress((i + 1) / len(gammas))
        barra.empty()

        fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))
        ax2.plot(gammas, rhos, marker='o', color="#1a1a1a", linewidth=2)
        ax2.axhline(0, color='grey', lw=0.7, linestyle='--')
        ax2.set_xlabel("γ  (0 = combinatorio · 0.5 = normalizado simétrico)")
        ax2.set_ylabel("Spearman(grado, V_γ)")
        ax2.set_title("Colapso de la anti-correlación al normalizar el operador")
        st.pyplot(fig2)

        caida = rhos[0] - rhos[-1]
        st.markdown(
            f"**De γ=0 a γ={gamma_max}: {rhos[0]:+.3f} → {rhos[-1]:+.3f}.**  \n" +
            ("Confirma dependencia fuerte del operador — la anti-correlación NO "
             "es una propiedad pura de esta topología." if abs(caida) > 0.3 else
             "En esta red el efecto es más estable frente al operador que en el "
             "promedio del estudio original — interesante caso a investigar.")
        )

# --- TAB 5: dinámica temporal ---
with tab5:
    st.markdown("### Autorretención: ¿qué tan rápido pierde un nodo su propia energía?")
    st.caption(
        "Se inyecta una perturbación en un solo nodo y se mide cuánta le queda "
        "a ese mismo nodo tras un tiempo T. V predice esto casi perfecto a T "
        "corto, y la predicción se degrada a tiempos largos."
    )
    if grande:
        st.warning("Red grande: este cálculo usa expm(-TL), puede tardar.")

    if st.button("▶️ Correr análisis de retención temporal"):
        Ts = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        rhos_t = []
        barra2 = st.progress(0)
        for i, T in enumerate(Ts):
            ET = expm(-L0 * T)
            ret = np.array([ET[j, j] ** 2 for j in range(N)])
            rho_t, _ = spearmanr(V, ret)
            rhos_t.append(rho_t)
            barra2.progress((i + 1) / len(Ts))
        barra2.empty()

        fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))
        ax3.plot(Ts, rhos_t, marker='o', color="#1a1a1a", linewidth=2)
        ax3.set_xscale("log")
        ax3.axhline(0, color='grey', lw=0.7, linestyle='--')
        ax3.set_xlabel("T (tiempo de difusión)")
        ax3.set_ylabel("Spearman(V, autorretención a tiempo T)")
        ax3.set_title("V predice bien a T corto; la predicción decae a T largo")
        st.pyplot(fig3)
        st.markdown(
            f"A T={Ts[0]}: Spearman={rhos_t[0]:+.3f}. A T={Ts[-1]}: "
            f"Spearman={rhos_t[-1]:+.3f}. Esto es consistente con que V, siendo "
            "una integral de 0 a ∞, está dominada por el comportamiento a "
            "tiempos cortos del sistema."
        )

# ----------------------------------------------------------------------
st.divider()
st.caption(
    "🕸️ SPG Explorer · Herramienta de exploración basada en la auditoría "
    "empírica publicada por Edher Alan Arteaga Marroquín "
    "(ORCID 0009-0004-7333-1975), DOI 10.5281/zenodo.21815650. "
    "No es una afirmación de que V sea un observable nuevo ni superior a las "
    "centralidades clásicas — ver el paper para el contexto y las limitaciones."
)
