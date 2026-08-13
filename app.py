"""
SPG Explorer — Spectral persistence analysis for complex networks
======================================================================
Based on: "A Systematic Empirical Evaluation of Spectral Persistence
Observables in Complex Networks" (Arteaga Marroquín, 2026)
DOI: 10.5281/zenodo.21815650

To run locally:
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
# Computation engine
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
# Interface
# ----------------------------------------------------------------------

st.title("🕸️ SPG Explorer")
st.markdown(
    "Analyze the **spectral persistence** of any complex network: how well a "
    "node retains a perturbation under diffusion, how that compares to degree "
    "and classical centralities, and how it depends on the diffusion operator "
    "chosen — the central finding of this work."
)

with st.expander("📖 What does this actually measure? — read before interpreting results"):
    st.markdown("""
The central observable, **V_i = Σ v_k(i)²/λ_k**, is mathematically identical to
**L⁺_ii**, the diagonal of the Laplacian pseudoinverse — a known object in
spectral graph theory (Van Mieghem, Devriendt & Cetinay, *Phys. Rev. E*
96, 032311, 2017), used there as a measure of a node's spreading capacity.

**Empirically verified across 30+ real networks, biological and non-biological:**
- In networks with heterogeneous degree, V anti-correlates strongly with degree.
- That anti-correlation **depends on the diffusion operator**, it is not fixed
  by the network alone: it collapses when moving from the combinatorial
  Laplacian to the symmetric normalized one.
- The effect is largely explained by V≈c/degree, though the fit quality varies
  substantially across networks (R² from 0.38 to 1.00 in the original study).
- τ̃ (a derived observable) was found redundant with V on community-structure
  tasks.

**This app does not claim V is a new observable**, nor that it outperforms
classical centralities in practical tasks (in fact, in a sensor-placement
benchmark, it did not — see paper, Section 8).

[Read the full paper (Zenodo, DOI)](https://doi.org/10.5281/zenodo.21815650) · 
[View code and notebooks (GitHub)](https://github.com/EdherAlanArteaga/spectral-persistence-audit)
""")

st.divider()

# --- mode selector ---
modo = st.radio(
    "Mode",
    ["🔍 Analyze one network", "⚖️ Compare multiple networks"],
    horizontal=True,
)
st.divider()

# ========================================================================
# MODE 2: COMPARE MULTIPLE NETWORKS
# ========================================================================
if modo == "⚖️ Compare multiple networks":
    st.subheader("Upload several networks to compare")
    st.markdown(
        "Same edge-list CSV format as single-network mode (two columns, no "
        "header, no self-loops). Upload two or more files — each is analyzed "
        "independently and the results are compared side by side."
    )
    archivos_multi = st.file_uploader(
        "CSV files", type=["csv"], accept_multiple_files=True, key="multi_uploader"
    )

    incluir_demo = st.checkbox("Also include the three built-in sample networks", value=False)

    resultados_multi = []
    fuentes = []
    if archivos_multi:
        for f in archivos_multi:
            try:
                df = pd.read_csv(f, comment='#', header=None)
                Gm = cargar_grafo(df)
                fuentes.append((Gm, f.name))
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")

    if incluir_demo:
        fuentes.append((nx.karate_club_graph(), "Karate Club"))
        Ger = nx.erdos_renyi_graph(150, 0.06, seed=1)
        Ger = Ger.subgraph(max(nx.connected_components(Ger), key=len)).copy()
        fuentes.append((Ger, "Erdős–Rényi (150n)"))
        fuentes.append((nx.barabasi_albert_graph(150, 2, seed=1), "Barabási–Albert (150n)"))

    if len(fuentes) < 2:
        st.warning("Upload at least two networks (or check the sample-networks box) to compare.")
        st.stop()

    with st.spinner(f"Computing spectral persistence for {len(fuentes)} networks…"):
        barra_multi = st.progress(0)
        filas_multi = []
        for i, (Gm, nombre) in enumerate(fuentes):
            Nm = Gm.number_of_nodes()
            if Nm < 4:
                st.warning(f"Skipping '{nombre}': fewer than 4 nodes.")
                barra_multi.progress((i + 1) / len(fuentes))
                continue
            Am = nx.to_numpy_array(Gm, nodelist=list(Gm.nodes()))
            degm = Am.sum(1)
            Lm = np.diag(degm) - Am
            Vm, evm, evecm, k0m = V_de_L(Lm)
            if Vm is None:
                st.warning(f"Skipping '{nombre}': no well-defined Fiedler mode.")
                barra_multi.progress((i + 1) / len(fuentes))
                continue
            sp_m, _ = spearmanr(degm, Vm)
            pe_m, _ = pearsonr(Vm, 1 / degm)
            dens_m = Am.sum() / (Nm * (Nm - 1))
            cv_m = degm.std() / degm.mean()

            # operator-dependence check at gamma=0.5 (fast, single point)
            Lg_half = L_gamma(Am, 0.5)
            Vg_half, _, _, _ = V_de_L(Lg_half)
            sp_half = spearmanr(Vg_half, degm)[0] if Vg_half is not None else np.nan

            filas_multi.append({
                "network": nombre,
                "nodes": Nm,
                "edges": Gm.number_of_edges(),
                "density": round(dens_m, 4),
                "degree CV": round(cv_m, 2),
                "Spearman(k,V) γ=0": round(sp_m, 3),
                "Spearman(k,V) γ=0.5": round(sp_half, 3) if not np.isnan(sp_half) else None,
                "R²(V, 1/k)": round(pe_m ** 2, 3),
            })
            barra_multi.progress((i + 1) / len(fuentes))
        barra_multi.empty()

    tabla_multi = pd.DataFrame(filas_multi)
    st.markdown("### Comparison table")
    st.dataframe(tabla_multi, use_container_width=True, hide_index=True)

    csv_multi = tabla_multi.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download comparison table (CSV)", data=csv_multi,
                        file_name="spg_network_comparison.csv", mime="text/csv")

    st.markdown("### Anti-correlation strength across networks")
    fig_multi, ax_multi = plt.subplots(figsize=(8, max(3, 0.4 * len(tabla_multi))))
    ax_multi.barh(tabla_multi["network"], tabla_multi["Spearman(k,V) γ=0"], color="#1a1a1a")
    ax_multi.axvline(0, color="grey", lw=0.7)
    ax_multi.set_xlabel("Spearman(degree, V), combinatorial Laplacian (γ=0)")
    st.pyplot(fig_multi)

    st.markdown("### Does the effect survive normalizing the operator (γ: 0 → 0.5)?")
    fig_multi2, ax_multi2 = plt.subplots(figsize=(8, max(3, 0.4 * len(tabla_multi))))
    x = np.arange(len(tabla_multi))
    w = 0.35
    ax_multi2.barh(x - w/2, tabla_multi["Spearman(k,V) γ=0"], height=w, label="γ=0 (combinatorial)", color="#1a1a1a")
    ax_multi2.barh(x + w/2, tabla_multi["Spearman(k,V) γ=0.5"].fillna(0), height=w, label="γ=0.5 (normalized)", color="#aaaaaa")
    ax_multi2.set_yticks(x); ax_multi2.set_yticklabels(tabla_multi["network"])
    ax_multi2.axvline(0, color="grey", lw=0.7)
    ax_multi2.legend()
    st.pyplot(fig_multi2)
    st.caption(
        "If the dark bars are much longer than the light bars for most "
        "networks, that reproduces the project's central finding: the "
        "anti-correlation depends heavily on the diffusion operator, not "
        "just on network topology."
    )

    st.divider()
    st.caption(
        "🕸️ SPG Explorer · Comparison mode. Based on the empirical audit "
        "published by Edher Alan Arteaga Marroquín (ORCID 0009-0004-7333-1975), "
        "DOI 10.5281/zenodo.21815650."
    )
    st.stop()

# ========================================================================
# MODE 1: SINGLE NETWORK ANALYSIS (unchanged below this point)
# ========================================================================

# --- data upload ---
st.subheader("1️⃣ Upload your network")

st.markdown(
    "This tool works with **any network that can be represented as a list of "
    "connections between two things** — it does not require data from any "
    "specific source. Examples of what fits: social networks (who follows "
    "whom), biological networks (protein–protein interactions, brain "
    "connectomes), infrastructure (airports linked by flights, computers "
    "linked by cables), citation networks, co-purchase networks, correlation "
    "networks thresholded into edges, and more."
)

with st.expander("📋 Required file format (click to see an example)"):
    st.markdown(
        "Your CSV needs **two columns**, one row per connection, **no header "
        "row**, no self-loops. Column names don't matter — only the first two "
        "columns are read. A third column (e.g. edge weight) is ignored in "
        "this version; the analysis treats every connection as present/absent."
    )
    st.markdown("**Example — a tiny 4-node network:**")
    ejemplo_df = pd.DataFrame(
        [["Ana", "Beto"], ["Beto", "Caro"], ["Caro", "Ana"], ["Caro", "Dana"]]
    )
    st.dataframe(ejemplo_df, use_container_width=False, hide_index=True,
                 column_config={"0": "node A", "1": "node B"})
    st.caption(
        "Node names can be text or numbers — anything pandas can read as a "
        "value works as a node identifier."
    )
    ejemplo_csv = ejemplo_df.to_csv(index=False, header=False).encode("utf-8")
    st.download_button("⬇️ Download this example as CSV", data=ejemplo_csv,
                        file_name="example_edges.csv", mime="text/csv")
    st.caption(
        "This format matches the `edges.csv` files from the Netzschleuder "
        "network repository (networks.skewed.de) exactly, but is not limited "
        "to that source — any two-column edge list works."
    )

col_up, col_demo = st.columns([2, 1])
with col_up:
    archivo = st.file_uploader("CSV file (up to 200MB)", type=["csv"])
with col_demo:
    demo_choice = st.selectbox(
        "…or use a sample network",
        ["(none)", "Karate Club (34 nodes)", "Erdős–Rényi random graph (150 nodes)",
         "Barabási–Albert / scale-free (150 nodes)"]
    )

G = None
if archivo is not None:
    try:
        df = pd.read_csv(archivo, comment='#', header=None)
        G = cargar_grafo(df)
        st.success(f"✅ Network loaded: **{G.number_of_nodes()}** nodes, "
                   f"**{G.number_of_edges()}** edges (largest connected component).")
    except Exception as e:
        st.error(f"Could not read the file: {e}")
elif demo_choice != "(none)":
    if "Karate" in demo_choice:
        G = nx.karate_club_graph()
    elif "Erdős" in demo_choice:
        G = nx.erdos_renyi_graph(150, 0.06, seed=1)
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    elif "Barabási" in demo_choice:
        G = nx.barabasi_albert_graph(150, 2, seed=1)
    st.info(f"Using sample network: {demo_choice} — {G.number_of_nodes()} nodes.")

if G is None:
    st.warning("Upload an edge-list CSV or pick a sample network to continue.")
    st.stop()

N = G.number_of_nodes()
if N < 4:
    st.error("The network needs at least 4 connected nodes.")
    st.stop()
grande = N > 800
if N > 3000:
    st.warning(f"This network has {N} nodes — computation may take a few minutes.")

nodes = list(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(1)
L0 = np.diag(deg) - A

with st.spinner("Computing spectrum and observables…"):
    V, ev, evec, k0 = V_de_L(L0)
    if V is None:
        st.error("This network does not have a well-defined Fiedler mode.")
        st.stop()
    tau_t = tau_tilde_de_L(V, ev, evec, k0)

sp_rho, _ = spearmanr(deg, V)
pe_rho, _ = pearsonr(V, 1 / deg)
r2 = pe_rho ** 2
density = A.sum() / (N * (N - 1))
cv_deg = deg.std() / deg.mean()

# ----------------------------------------------------------------------
# Summary panel
# ----------------------------------------------------------------------
st.divider()
st.subheader("2️⃣ Network summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", N)
c2.metric("Edges", G.number_of_edges())
c3.metric("Density", f"{density:.4f}")
c4.metric("Degree CV", f"{cv_deg:.2f}")
c5.metric("Spearman(k, V)", f"{sp_rho:+.3f}")

if abs(sp_rho) > 0.7:
    veredicto = "🔴 Strong anti-centrality: higher-degree nodes tend to retain less."
elif abs(sp_rho) > 0.3:
    veredicto = "🟡 Moderate anti-centrality."
else:
    veredicto = "⚪ No clear anti-centrality — possibly a network with homogeneous degree."
st.markdown(f"**Verdict:** {veredicto}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Degree vs. V", "🏆 Node table", "🔬 Vs. classical centralities",
    "🧭 Operator dependence", "🌊 Temporal dynamics"
])

# --- TAB 1: degree vs V ---
with tab1:
    st.markdown("### Degree vs. spectral persistence")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))
    ax1.scatter(deg, V, alpha=0.6, s=28, color="#1a1a1a")
    ax1.set_xlabel("degree"); ax1.set_ylabel("V (= L⁺_ii)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_title("Log-log scale")
    st.pyplot(fig1)

    st.markdown("### How much of this is simple 1/degree scaling?")
    fig1b, ax1b = plt.subplots(figsize=(6.5, 4.8))
    ax1b.scatter(1/deg, V, alpha=0.6, s=28, color="#555555")
    ax1b.set_xlabel("1 / degree"); ax1b.set_ylabel("V")
    ax1b.set_title(f"R² = {r2:.3f}")
    st.pyplot(fig1b)
    st.caption(
        f"If R² is close to 1.0, V is almost entirely explained by the "
        f"trivial scaling V≈c/degree. In the original study, this value "
        f"ranged from 0.38 to 1.00 across 11 real networks, with density as "
        f"the best partial predictor found (a moderate correlation, not a "
        f"clean law)."
    )

# --- TAB 2: node table ---
with tab2:
    st.markdown("### Nodes ranked by persistence (V)")
    tabla = pd.DataFrame({
        "node": nodes, "degree": deg.astype(int), "V": V, "τ̃": tau_t,
    }).sort_values("V", ascending=False).reset_index(drop=True)
    tabla.index += 1

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Top 10 — highest persistence (best 'retainers')**")
        st.dataframe(tabla.head(10), use_container_width=True)
    with colB:
        st.markdown("**Bottom 10 — lowest persistence (best 'spreaders')**")
        st.dataframe(tabla.tail(10).sort_values("V"), use_container_width=True)

    st.markdown("### Full table")
    st.dataframe(tabla, use_container_width=True, height=350)

    csv_out = tabla.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download full table (CSV)", data=csv_out,
                        file_name="spg_nodes.csv", mime="text/csv")

# --- TAB 3: vs classical centralities ---
with tab3:
    st.markdown("### V vs. classical centralities")
    if grande:
        st.warning(
            "Large network (>800 nodes): betweenness and current-flow "
            "closeness can be slow. Enable if you have time to wait."
        )
        correr_completo = st.checkbox("Compute anyway (may take a while)")
    else:
        correr_completo = True

    if correr_completo:
        with st.spinner("Computing classical centralities…"):
            eig_c = nx.eigenvector_centrality_numpy(G)
            clo_c = nx.closeness_centrality(G)
            try:
                cfc_c = nx.current_flow_closeness_centrality(G)
            except Exception:
                cfc_c = {n: np.nan for n in nodes}
            btw_c = nx.betweenness_centrality(G) if N <= 2000 else None

        comp = pd.DataFrame({
            "degree": deg,
            "V": V,
            "eigenvector": [eig_c[n] for n in nodes],
            "closeness": [clo_c[n] for n in nodes],
            "current_flow_closeness": [cfc_c[n] for n in nodes],
        })
        if btw_c is not None:
            comp["betweenness"] = [btw_c[n] for n in nodes]

        st.markdown("**Rank correlation (Spearman) of V against each metric:**")
        filas = []
        for col in comp.columns:
            if col == "V":
                continue
            rho, _ = spearmanr(V, comp[col])
            filas.append({"metric": col, "Spearman(V, ·)": round(rho, 3)})
        tabla_corr = pd.DataFrame(filas).sort_values("Spearman(V, ·)", key=abs, ascending=False)
        st.dataframe(tabla_corr, use_container_width=True, hide_index=True)

        cfc_rho = tabla_corr.loc[tabla_corr["metric"] == "current_flow_closeness", "Spearman(V, ·)"]
        if len(cfc_rho) and abs(cfc_rho.values[0]) > 0.95:
            st.info(
                "V ranks nodes almost identically to *current-flow closeness "
                "centrality* — they are strongly related but **not the same "
                "quantity** (they differ in exact value, not just rank order; "
                "see paper, Section 4)."
            )
    else:
        st.info("Enable the checkbox to compute classical centralities.")

# --- TAB 4: operator dependence ---
with tab4:
    st.markdown("### Is the anti-correlation about the network, or the diffusion operator?")
    st.caption(
        "L_γ = D⁻ᵞ(D−A)D⁻ᵞ. γ=0 → combinatorial Laplacian (diagonal = degree). "
        "γ=0.5 → symmetric normalized Laplacian (diagonal uniformly = 1). "
        "**This is the strongest finding of the original project.**"
    )
    gamma_max = st.slider("Compute up to γ =", 0.1, 0.5, 0.5, 0.05, key="gamma_slider")
    n_pasos = st.slider("Resolution (number of points)", 5, 25, 12)

    if st.button("▶️ Run γ sweep"):
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
        ax2.set_xlabel("γ  (0 = combinatorial · 0.5 = symmetric normalized)")
        ax2.set_ylabel("Spearman(degree, V_γ)")
        ax2.set_title("Collapse of the anti-correlation as the operator is normalized")
        st.pyplot(fig2)

        caida = rhos[0] - rhos[-1]
        st.markdown(
            f"**From γ=0 to γ={gamma_max}: {rhos[0]:+.3f} → {rhos[-1]:+.3f}.**  \n" +
            ("Confirms strong operator dependence — the anti-correlation is NOT "
             "a pure property of this topology." if abs(caida) > 0.3 else
             "In this network the effect is more stable against the operator "
             "than the average of the original study — an interesting case to "
             "investigate further.")
        )

# --- TAB 5: temporal dynamics ---
with tab5:
    st.markdown("### Self-retention: how fast does a node lose its own energy?")
    st.caption(
        "A perturbation is injected at a single node, and we measure how much "
        "of it remains at that same node after time T. V predicts this almost "
        "perfectly at short T, and the prediction degrades at long T."
    )
    if grande:
        st.warning("Large network: this computation uses expm(-TL), it may be slow.")

    if st.button("▶️ Run temporal retention analysis"):
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
        ax3.set_xlabel("T (diffusion time)")
        ax3.set_ylabel("Spearman(V, self-retention at time T)")
        ax3.set_title("V predicts well at short T; the prediction decays at long T")
        st.pyplot(fig3)
        st.markdown(
            f"At T={Ts[0]}: Spearman={rhos_t[0]:+.3f}. At T={Ts[-1]}: "
            f"Spearman={rhos_t[-1]:+.3f}. This is consistent with V, being an "
            "integral from 0 to ∞, being dominated by the system's short-time "
            "behavior."
        )

# ----------------------------------------------------------------------
st.divider()
st.caption(
    "🕸️ SPG Explorer · Exploration tool based on the empirical audit "
    "published by Edher Alan Arteaga Marroquín "
    "(ORCID 0009-0004-7333-1975), DOI 10.5281/zenodo.21815650. "
    "This is not a claim that V is a new observable or superior to classical "
    "centralities — see the paper for full context and limitations."
)
