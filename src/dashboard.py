"""
Dashboard Streamlit pour la maintenance prédictive MECHA.

Fournit aux équipes métiers (maintenance, production, pilotage) :
- Vue globale de l'état des machines et des usines
- Prédictions de pannes et indicateurs de risque (via l'API REST)
- Historique des données capteurs par machine
- Tableaux de bord de performance des modèles
- Monitoring temps réel et gestion des alertes (via l'API REST)

Toutes les prédictions passent par l'API (variable API_URL) : le dashboard ne
charge aucun modèle. Seuils, couleurs et fréquence de rafraîchissement
proviennent de config/regles_metier.yaml.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:          # streamlit run src/dashboard.py
    sys.path.insert(0, str(BASE_DIR))
from src import regles  # noqa: E402

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
API_URL       = os.getenv("API_URL", "http://localhost:8000")

REGLES = regles.charger_regles()

SENSOR_COLS = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar",  "vitesse_tr_min",
]

SENSOR_LABELS = {
    "temperature_C":   "Température (°C)",
    "vibration_mm_s":  "Vibration (mm/s)",
    "courant_A":       "Courant (A)",
    "pression_bar":    "Pression (bar)",
    "vitesse_tr_min":  "Vitesse (tr/min)",
}

ETAT_COLORS = {
    "normal":    "#28a745",
    "a_risque":  "#ffc107",
    "critique":  "#fd7e14",
    "en_panne":  "#dc3545",
}

# Couleurs des niveaux de risque (panne 24 h) et d'anomalie
RISK_COLORS = {
    **regles.couleurs_niveaux(REGLES),
    "anomalie": "#dc3545",
    "suspect":  "#fd7e14",
    "normal":   "#28a745",
}

NUMERIC_FEATURES = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar", "vitesse_tr_min",
    "age_machine_h", "h_depuis_maintenance",
]

FEATURE_LABELS = {
    **SENSOR_LABELS,
    "age_machine_h":        "Âge machine (h)",
    "h_depuis_maintenance": "Heures depuis maintenance",
    "type_machine_encoded": "Type de machine",
    "usine_encoded":        "Usine",
}

st.set_page_config(
    page_title="MECHA - Maintenance Prédictive",
    page_icon=":factory:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    path = PROCESSED_DIR / "mecha_unified_prepared.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return None


@st.cache_data
def load_json(path: Path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_training_results():
    return load_json(MODELS_DIR / "training_results.json")


@st.cache_data
def load_category_maps():
    maps = load_json(PROCESSED_DIR / "category_maps.json")
    return maps or {"type_machine": {}, "usine_id": {}}


# ---------------------------------------------------------------------------
# Accès à l'API
# ---------------------------------------------------------------------------

def api_get(path: str, **params):
    """GET sur l'API. Retourne (données, None) ou (None, message d'erreur)."""
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, timeout=10.0)
    except httpx.RequestError as exc:
        return None, f"API injoignable ({API_URL}) : {exc}"
    if r.status_code != 200:
        return None, f"API {path} — HTTP {r.status_code} : {r.text[:200]}"
    return r.json(), None


def api_post(path: str, payload: dict | None = None, timeout: float = 30.0):
    """POST sur l'API. Retourne (données, None) ou (None, message d'erreur)."""
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        return None, f"API injoignable ({API_URL}) : {exc}"
    if r.status_code != 200:
        return None, f"API {path} — HTTP {r.status_code} : {r.text[:300]}"
    return r.json(), None


# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------

def gauge_steps() -> list[dict]:
    """Zones colorées des jauges, construites à partir des seuils du fichier de règles."""
    steps, haut = [], 100
    for niveau, seuil in regles.seuils_panne(REGLES):
        steps.append({"range": [seuil * 100, haut], "color": RISK_COLORS[niveau]})
        haut = seuil * 100
    return steps


def jauge(probabilite: float, titre: str, hauteur: int | None = None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probabilite * 100, 1),
        title={"text": titre},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkblue"}, "steps": gauge_steps()},
    ))
    if hauteur:
        fig.update_layout(height=hauteur, margin=dict(t=50, b=0, l=0, r=0))
    return fig


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def regle_configuree(results: dict | None) -> dict | None:
    """Ligne de la validation correspondant à la règle d'alerte du fichier de config."""
    if not results or "validation_regle_alerte" not in results:
        return None
    for ligne in results["validation_regle_alerte"]["candidats"]:
        if ligne.get("regle_configuree"):
            return ligne
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("MECHA")
st.sidebar.markdown("### Maintenance Prédictive IA")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d'ensemble",
        "Prédiction en temps réel",
        "Analyse des capteurs",
        "Performance des modèles",
        "Monitoring Temps Réel",
    ],
)
st.sidebar.caption(f"API : `{API_URL}`")


# ---------------------------------------------------------------------------
# Page 1 : Vue d'ensemble
# ---------------------------------------------------------------------------

if page == "Vue d'ensemble":
    st.title("Tableau de bord — Vue d'ensemble")

    df = load_data()
    if df is None:
        st.warning("Aucune donnée trouvée. Lancez d'abord `python src/data_preparation.py`.")
        st.stop()

    results = load_training_results()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Usines", df["usine_id"].nunique())
    col2.metric("Machines", df["machine_id"].nunique())
    col3.metric("Relevés horaires", f"{len(df):,}")
    col4.metric("Taux en panne", f"{df['en_panne'].mean():.2%}")

    regle = regle_configuree(results)
    if regle:
        st.markdown("#### Performance de la règle d'alerte (validation sur la période la plus récente)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pannes anticipées (24 h avant)", pct(regle["rappel_evenements"]))
        k2.metric("Alertes suivies d'une panne sous 7 j",
                  pct(regle["part_alertes_suivies_de_panne"].get("168h")))
        k3.metric("Alertes suivies d'une panne sous 24 h",
                  pct(regle["part_alertes_suivies_de_panne"].get("24h")))
        k4.metric("Alertes / machine / semaine", f"{regle['alertes_par_machine_semaine']:.2f}")

    qualite = load_json(PROCESSED_DIR / "rapport_qualite.json")
    if qualite:
        with st.expander("Qualité des données (nettoyage du fichier brut)"):
            st.dataframe(
                pd.DataFrame(list(qualite.items()), columns=["Indicateur", "Valeur"]),
                width="stretch", hide_index=True,
            )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("État des machines par usine")
        etat_usine = (
            df.groupby(["usine_id", "etat_machine"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            etat_usine, x="usine_id", y="count", color="etat_machine",
            title="Distribution des états par usine",
            color_discrete_map=ETAT_COLORS,
            barmode="stack",
        )
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Distribution globale des états")
        etat_dist = df["etat_machine"].value_counts().reset_index()
        etat_dist.columns = ["État", "Count"]
        fig2 = px.pie(
            etat_dist, names="État", values="Count",
            title="Répartition des états machines",
            color="État", color_discrete_map=ETAT_COLORS,
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    st.subheader("Top 10 machines — heures en panne")
    pannes_machine = (
        df.groupby("machine_id")["en_panne"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    pannes_machine.columns = ["Machine", "Heures en panne"]
    fig3 = px.bar(
        pannes_machine, x="Machine", y="Heures en panne",
        color="Heures en panne", color_continuous_scale="Reds",
    )
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Évolution des pannes dans le temps")
    df_daily = (
        df.set_index("timestamp")
        .resample("D")["en_panne"]
        .sum()
        .reset_index()
    )
    df_daily.columns = ["Date", "Pannes"]
    fig4 = px.line(df_daily, x="Date", y="Pannes",
                   title="Nombre d'heures en panne par jour")
    st.plotly_chart(fig4, width="stretch")


# ---------------------------------------------------------------------------
# Page 2 : Prédiction en temps réel
# ---------------------------------------------------------------------------

elif page == "Prédiction en temps réel":
    st.title("Prédiction de panne en temps réel")

    maps = load_category_maps()
    types_machine = list(maps["type_machine"].keys()) or \
                    ["CNC-Fraisage", "CNC-Tournage", "Découpe-Laser", "Centre-Usinage"]
    usines = list(maps["usine_id"].keys()) or \
             ["USN-FR-01", "USN-FR-02", "USN-FR-03", "USN-ES-01", "USN-ES-02"]

    tab_manuel, tab_csv = st.tabs(["Saisie manuelle", "Prédiction par CSV"])

    # -----------------------------------------------------------------------
    # Onglet 1 : saisie manuelle
    # -----------------------------------------------------------------------
    with tab_manuel:
        st.subheader("Paramètres de la machine")
        col1, col2, col3 = st.columns(3)

        with col1:
            temperature  = st.slider("Température (°C)",    30.0, 200.0, 70.0, 0.5)
            vibration    = st.slider("Vibration (mm/s)",     0.0,  20.0,  2.5, 0.05)
            courant      = st.slider("Courant (A)",           5.0,  60.0, 20.0, 0.1)

        with col2:
            pression     = st.slider("Pression (bar)",        0.0,  10.0,  6.0, 0.05)
            vitesse      = st.slider("Vitesse (tr/min)",       0,   2000, 1400, 10)
            age          = st.slider("Âge machine (h)",        0, 100000, 10000, 100)

        with col3:
            h_maint      = st.slider("H depuis maintenance", 0, 5000, 300, 10)
            type_machine = st.selectbox("Type machine", types_machine)
            usine_id     = st.selectbox("Usine", usines)

        if st.button("Prédire", type="primary"):
            payload = {
                "temperature_C": temperature, "vibration_mm_s": vibration,
                "courant_A": courant, "pression_bar": pression,
                "vitesse_tr_min": vitesse, "age_machine_h": age,
                "h_depuis_maintenance": h_maint,
                "type_machine": type_machine, "usine_id": usine_id,
            }
            res, err = api_post("/predict/panne24h/explain", payload)
            if err:
                st.error(err)
            else:
                rul, err_rul = api_post("/predict/rul", payload)
                couleur = RISK_COLORS.get(res["risk_level"], "#6c757d")

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Panne dans 24h", "OUI" if res["prediction"] == 1 else "NON")
                c2.metric("Probabilité", f"{res['probability']:.1%}")
                c3.markdown(
                    f"**Risque :** <span style='color:{couleur};font-weight:bold'>"
                    f"{res['risk_level'].upper()}</span><br>{res['recommendation']}",
                    unsafe_allow_html=True,
                )
                if rul:
                    c4.metric("RUL estimé (ordre de grandeur)", f"{rul['estimated_rul_hours']:.0f} h")
                elif err_rul:
                    c4.caption(err_rul)

                g1, g2 = st.columns([3, 2])
                with g1:
                    st.plotly_chart(jauge(res["probability"], "Probabilité de panne dans 24h (%)"),
                                    width="stretch")
                with g2:
                    st.markdown("**Pourquoi cette prédiction ?** "
                                f"(_{res['methode'].replace('_', ' ')}_)")
                    expl = pd.DataFrame(res["top_features"])
                    expl["facteur"] = expl["feature"].map(FEATURE_LABELS).fillna(expl["feature"])
                    fig_e = px.bar(
                        expl, x="contribution", y="facteur", orientation="h",
                        color="direction",
                        color_discrete_map={"hausse_risque": "#dc3545", "baisse_risque": "#28a745"},
                        hover_data=["value"],
                    )
                    fig_e.update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0),
                                        yaxis_title=None, legend_title=None)
                    st.plotly_chart(fig_e, width="stretch")
                    st.caption("Contribution de chaque mesure à la probabilité de panne : "
                               "en rouge elle l'augmente, en vert elle la diminue.")

    # -----------------------------------------------------------------------
    # Onglet 2 : prédiction par CSV (endpoint /predict/batch)
    # -----------------------------------------------------------------------
    with tab_csv:
        st.subheader("Rapport de pannes potentielles via CSV")

        SAMPLE_COLS = ["machine_id", "type_machine", "usine_id"] + NUMERIC_FEATURES

        st.markdown("#### Télécharger un CSV exemple")
        df_full = load_data()
        if df_full is not None:
            pannes  = df_full[df_full["panne_dans_24h"] == 1][SAMPLE_COLS + ["panne_dans_24h"]].sample(
                n=min(50, int((df_full["panne_dans_24h"] == 1).sum())), random_state=42
            )
            normaux = df_full[df_full["panne_dans_24h"] == 0][SAMPLE_COLS + ["panne_dans_24h"]].sample(
                n=50, random_state=42
            )
            sample_csv = pd.concat([pannes, normaux]).sample(frac=1, random_state=42).reset_index(drop=True)
            col_dl, col_prev = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="Télécharger le CSV exemple (100 lignes)",
                    data=sample_csv.to_csv(index=False).encode("utf-8"),
                    file_name="mecha_sample.csv",
                    mime="text/csv",
                )
                st.caption(f"Colonnes obligatoires : `{', '.join(NUMERIC_FEATURES)}`")
                st.caption("Colonnes optionnelles : `machine_id, type_machine, usine_id`")
            with col_prev:
                with st.expander("Aperçu"):
                    st.dataframe(sample_csv.head(5), width="stretch")
        else:
            st.info("Données non disponibles pour générer le CSV exemple.")

        st.markdown("---")

        uploaded = st.file_uploader("Uploader un CSV pour générer le rapport", type=["csv"])

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)

                missing = [c for c in NUMERIC_FEATURES if c not in df_upload.columns]
                if missing:
                    st.error(f"Colonnes manquantes dans le CSV : {missing}")
                    st.stop()

                # Type et usine : noms attendus par l'API (rétro-compatibilité avec les colonnes encodées)
                inv_type  = {v: k for k, v in maps["type_machine"].items()}
                inv_usine = {v: k for k, v in maps["usine_id"].items()}
                if "type_machine" not in df_upload.columns:
                    df_upload["type_machine"] = (df_upload["type_machine_encoded"].map(inv_type)
                                                 if "type_machine_encoded" in df_upload.columns
                                                 else types_machine[0])
                if "usine_id" not in df_upload.columns:
                    df_upload["usine_id"] = (df_upload["usine_encoded"].map(inv_usine)
                                             if "usine_encoded" in df_upload.columns
                                             else usines[0])
                if "machine_id" not in df_upload.columns:
                    df_upload.insert(0, "machine_id", [f"Machine-{i+1}" for i in range(len(df_upload))])

                # Contrôle de plausibilité : mêmes règles que l'API
                bornes = REGLES["capteurs"]
                valide = df_upload[NUMERIC_FEATURES].notna().all(axis=1)
                for col in NUMERIC_FEATURES:
                    valide &= df_upload[col].between(bornes[col]["min"], bornes[col]["max"])
                valide &= df_upload["type_machine"].isin(types_machine) & df_upload["usine_id"].isin(usines)
                n_ignores = int((~valide).sum())
                df_ok = df_upload[valide].reset_index(drop=True)
                if n_ignores:
                    st.warning(f"{n_ignores} ligne(s) ignorée(s) : valeur manquante, hors bornes "
                               "physiques ou type/usine inconnu.")
                if df_ok.empty:
                    st.stop()

                lignes = [
                    {**{c: float(r[c]) for c in NUMERIC_FEATURES[:5]},
                     "age_machine_h": int(r["age_machine_h"]),
                     "h_depuis_maintenance": int(r["h_depuis_maintenance"]),
                     "type_machine": r["type_machine"], "usine_id": r["usine_id"]}
                    for _, r in df_ok.iterrows()
                ]
                resultats = []
                for i in range(0, len(lignes), 500):
                    rep, err = api_post("/predict/batch", {"lignes": lignes[i:i + 500]}, timeout=120)
                    if err:
                        st.error(err)
                        st.stop()
                    resultats.extend(rep["resultats"])

                df_result = df_ok.copy()
                df_result["panne_dans_24h_predite"] = [r["panne24h"]["prediction"] for r in resultats]
                df_result["probabilite"] = [r["panne24h"]["probability"] for r in resultats]
                df_result["risque"]      = [r["panne24h"]["risk_level"] for r in resultats]
                df_result["action"]      = [r["panne24h"]["recommendation"] for r in resultats]
                has_rul = all(r.get("rul") for r in resultats)
                if has_rul:
                    df_result["rul_estime_h"] = [round(r["rul"]["estimated_rul_hours"], 1) for r in resultats]

                st.success(f"{len(df_result)} lignes analysées.")

                # A — Résumé
                seuil_alerte = min(s for n, s in regles.seuils_panne(REGLES) if s > 0)
                niveau_max = regles.seuils_panne(REGLES)[0][0]
                a_risque = df_result["probabilite"] >= seuil_alerte
                nb_risque   = int(a_risque.sum())
                nb_critique = int((df_result["risque"] == niveau_max).sum())
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lignes analysées", len(df_result))
                k2.metric("Machines à risque", nb_risque)
                k3.metric(f"Dont {niveau_max}", nb_critique)
                if has_rul and nb_risque > 0:
                    k4.metric("RUL moyen (à risque)",
                              f"{df_result.loc[a_risque, 'rul_estime_h'].mean():.0f} h")

                st.markdown("---")

                # B — Rapport des pannes potentielles
                st.subheader("Rapport des pannes potentielles")
                df_risque = df_result[a_risque].sort_values("probabilite", ascending=False) \
                                               .reset_index(drop=True)
                if df_risque.empty:
                    st.success("Aucune machine à risque détectée dans ce fichier.")
                else:
                    rapport_cols = ["machine_id", "type_machine", "usine_id", "probabilite", "risque"]
                    if has_rul:
                        rapport_cols.append("rul_estime_h")
                    rapport_cols.append("action")

                    df_rapport = df_risque[rapport_cols].copy()
                    df_rapport["probabilite"] = (df_rapport["probabilite"] * 100).round(1).astype(str) + " %"
                    st.dataframe(df_rapport, width="stretch", hide_index=True)
                    st.download_button(
                        label=f"Télécharger le rapport ({len(df_risque)} machines à risque)",
                        data=df_risque[rapport_cols].to_csv(index=False).encode("utf-8"),
                        file_name="rapport_pannes_mecha.csv",
                        mime="text/csv",
                    )

                st.markdown("---")

                # C — Visualisations
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    risk_counts = df_result["risque"].value_counts().reset_index()
                    risk_counts.columns = ["risque", "count"]
                    fig_pie = px.pie(
                        risk_counts, values="count", names="risque",
                        title="Répartition par niveau de risque",
                        color="risque", color_discrete_map=RISK_COLORS,
                    )
                    st.plotly_chart(fig_pie, width="stretch")
                with col_viz2:
                    if has_rul and not df_risque.empty:
                        df_rul_chart = df_risque.nsmallest(20, "rul_estime_h")[
                            ["machine_id", "rul_estime_h", "risque"]
                        ].sort_values("rul_estime_h")
                        fig_rul = px.bar(
                            df_rul_chart, x="machine_id", y="rul_estime_h",
                            color="risque", color_discrete_map=RISK_COLORS,
                            title="RUL des 20 machines les plus urgentes (heures restantes)",
                            labels={"rul_estime_h": "RUL (h)", "machine_id": "Machine"},
                        )
                        fig_rul.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_rul, width="stretch")

                st.markdown("---")

                # D — Données complètes
                with st.expander("Données complètes (toutes les lignes)"):
                    st.dataframe(df_result, width="stretch")
                    st.download_button(
                        label="Télécharger le CSV complet avec prédictions",
                        data=df_result.to_csv(index=False).encode("utf-8"),
                        file_name="mecha_predictions_completes.csv",
                        mime="text/csv",
                    )

                # E — Comparaison avec les labels réels (si présents)
                label_col = next((c for c in ("panne_dans_24h", "en_panne") if c in df_ok.columns), None)
                if label_col:
                    from sklearn.metrics import classification_report
                    report = classification_report(
                        df_ok[label_col].values, df_result["panne_dans_24h_predite"].values,
                        output_dict=True, zero_division=0,
                    )
                    st.subheader(f"Comparaison avec les labels réels ({label_col})")
                    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

            except Exception as e:
                st.error(f"Erreur lors du traitement : {e}")


# ---------------------------------------------------------------------------
# Page 3 : Analyse des capteurs
# ---------------------------------------------------------------------------

elif page == "Analyse des capteurs":
    st.title("Analyse des données capteurs")

    df = load_data()
    if df is None:
        st.warning("Données non disponibles.")
        st.stop()

    machine_ids = sorted(df["machine_id"].unique())
    selected_machine = st.selectbox("Sélectionner une machine", machine_ids)
    machine_df = df[df["machine_id"] == selected_machine].sort_values("timestamp")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Type",  machine_df["type_machine"].iloc[0])
        st.metric("Usine", machine_df["usine_id"].iloc[0])
    with col_b:
        st.metric("Âge actuel (h)", f"{machine_df['age_machine_h'].iloc[-1]:,}")
        st.metric("Pannes totales", int(machine_df["nb_pannes_total"].iloc[-1]))

    st.markdown("---")

    st.subheader(f"Capteurs — {selected_machine} (500 dernières mesures)")
    tail = machine_df.tail(500)

    for col in SENSOR_COLS:
        fig = px.line(tail, x="timestamp", y=col, title=SENSOR_LABELS[col])
        pannes = tail[tail["en_panne"] == 1]
        if not pannes.empty:
            fig.add_trace(go.Scatter(
                x=pannes["timestamp"], y=pannes[col],
                mode="markers",
                marker=dict(color="red", size=4),
                name="En panne",
            ))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Corrélation capteurs × score_degradation")
    corr_cols = SENSOR_COLS + ["score_degradation", "rul_heures", "en_panne"]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", title="Matrice de corrélation",
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig_corr, width="stretch")

    st.subheader("Distribution des capteurs par état machine")
    selected_sensor = st.selectbox("Capteur", SENSOR_COLS,
                                   format_func=lambda x: SENSOR_LABELS[x])
    fig_dist = px.box(
        df, x="etat_machine", y=selected_sensor,
        color="etat_machine", color_discrete_map=ETAT_COLORS,
        title=f"{SENSOR_LABELS[selected_sensor]} par état machine",
        category_orders={"etat_machine": ["normal", "a_risque", "critique", "en_panne"]},
    )
    st.plotly_chart(fig_dist, width="stretch")


# ---------------------------------------------------------------------------
# Page 4 : Performance des modèles
# ---------------------------------------------------------------------------

elif page == "Performance des modèles":
    st.title("Performance des modèles de Machine Learning")

    results = load_training_results()
    if results is None:
        st.warning("Aucun résultat trouvé. Lancez d'abord `python src/model_training.py`.")
        st.stop()

    protocole = results.get("protocole", {})
    METRIQUES = ("accuracy", "precision", "recall", "f1_score", "auc_roc")
    NOMS = {"accuracy": "Accuracy", "precision": "Précision", "recall": "Rappel",
            "f1_score": "F1", "auc_roc": "AUC-ROC"}

    def nom_modele(cle: str) -> str:
        return cle.replace("_", " ").capitalize()

    for task_key, label in [
        ("classification_panne_24h", "Classification — panne dans 24 h (modèle des alertes)"),
        ("classification_en_panne",  "Classification — machine en panne"),
    ]:
        task = results.get(task_key)
        if not task:
            continue
        st.subheader(label)
        if protocole.get("classification"):
            st.caption(protocole["classification"])

        t5050, treel = st.tabs(["Test 50/50 — sait-il reconnaître une panne ?",
                                "Distribution réelle — ce que verra la maintenance"])
        with t5050:
            rows = [{"Modèle": nom_modele(m), **{NOMS[k]: v[k] for k in METRIQUES}}
                    for m, v in task.items()]
            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df.round(3), width="stretch", hide_index=True)
            fig = go.Figure()
            for _, row in metrics_df.iterrows():
                fig.add_trace(go.Bar(name=row["Modèle"], x=list(NOMS.values()),
                                     y=[row[n] for n in NOMS.values()]))
            fig.update_layout(barmode="group", yaxis_range=[0, 1],
                              title="Comparaison des modèles (test 50/50)")
            st.plotly_chart(fig, width="stretch")
            st.caption("Le modèle naïf répond toujours « pas de panne » : 50 % d'accuracy ici, "
                       "mais ~97 % sur les données réelles où les pannes sont rares.")
        with treel:
            rows = []
            for m, v in task.items():
                r = v.get("distribution_reelle")
                if not r:
                    continue
                niveaux = r["alertes"]["niveaux"]
                row = {"Modèle": nom_modele(m), **{NOMS[k]: r[k] for k in METRIQUES}}
                for niv, d in niveaux.items():
                    row[f"Fausses alertes « {niv} »"] = d["taux_fausses_alertes"]
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
                st.caption("Calculé relevé par relevé, au taux de pannes réel. Voir plus bas la "
                           "validation de la règle d'alerte, qui raisonne par événement.")
        st.markdown("---")

    reg = results.get("regression_rul")
    if reg:
        st.subheader("Régression — temps restant avant panne (RUL)")
        if protocole.get("regression_rul"):
            st.caption(protocole["regression_rul"])
        st.dataframe(
            pd.DataFrame([{"Modèle": nom_modele(m), "MAE (h)": v["mae"], "RMSE (h)": v["rmse"],
                           "R²": v["r2"]} for m, v in reg.items()]).round(3),
            width="stretch", hide_index=True,
        )
        st.markdown("---")

    ano = results.get("anomaly_detection")
    if ano:
        st.subheader("Détection d'anomalies — Isolation Forest")
        st.dataframe(pd.DataFrame([ano]).rename(columns=lambda c: c.replace("_", " ").capitalize()),
                     width="stretch", hide_index=True)
        st.markdown("---")

    val = results.get("validation_regle_alerte")
    if val:
        st.subheader("Validation de la règle d'alerte")
        st.caption(val["protocole"])
        rows = [{
            "Seuil": l["seuil"],
            "Lectures consécutives": l["lectures_consecutives"],
            "Règle configurée": "✔" if l["regle_configuree"] else "",
            "Alertes": l["n_alertes"],
            "Pannes anticipées (24 h)": l["rappel_evenements"],
            "Alertes suivies d'une panne sous 24 h": l["part_alertes_suivies_de_panne"]["24h"],
            "… sous 72 h": l["part_alertes_suivies_de_panne"]["72h"],
            "… sous 7 j": l["part_alertes_suivies_de_panne"]["168h"],
            "Alertes / machine / semaine": l["alertes_par_machine_semaine"],
            "Anticipation médiane (h)": l["anticipation_mediane_h"],
        } for l in val["candidats"]]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        st.markdown("---")

    st.subheader("Visualisations sauvegardées")
    imgs = sorted(MODELS_DIR.glob("*.png"))
    if imgs:
        for img in imgs:
            st.image(str(img), caption=img.stem.replace("_", " ").title())
    else:
        st.info("Aucune image. Lancez l'entraînement pour générer les graphiques.")


# ---------------------------------------------------------------------------
# Page 5 : Monitoring Temps Réel (PostgreSQL via l'API)
# ---------------------------------------------------------------------------

elif page == "Monitoring Temps Réel":
    rafraichissement = int(REGLES["frequences"]["dashboard_rafraichissement_s"])
    st.title("Monitoring Temps Réel")
    st.caption(
        f"Données en direct depuis PostgreSQL via `{API_URL}` — "
        f"rafraîchissement auto toutes les {rafraichissement} secondes"
    )

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        limit = st.slider("Nombre de prédictions à afficher", 10, 200, 50, step=10)
    with col_ctrl2:
        auto_refresh = st.checkbox(f"Auto-refresh ({rafraichissement}s)", value=True)

    recent,  err1 = api_get("/monitoring/recent", limit=limit)
    summary, err2 = api_get("/monitoring/machines")
    alertes, err3 = api_get("/alertes", statut="ouverte", limit=100)

    if err1 or err2:
        st.warning(
            "Impossible de joindre l'API ou la base de données. "
            "Vérifiez que PostgreSQL est démarré et que le simulateur tourne.\n\n"
            f"{err1 or err2}"
        )
        st.stop()

    # --- Alertes ouvertes ---
    st.subheader("Alertes ouvertes")
    regle_panne = REGLES["alertes"]["panne24h"]
    st.caption(
        f"Règle : niveau {', '.join(regle_panne['niveaux'])} sur "
        f"{regle_panne['lectures_consecutives']} lectures consécutives "
        "(config/regles_metier.yaml). Acquitter une alerte quand l'intervention est prise en charge."
    )
    if err3:
        st.info(err3)
    elif not alertes["alertes"]:
        st.success("Aucune alerte ouverte.")
    else:
        for a in alertes["alertes"]:
            c1, c2, c3, c4 = st.columns([2, 2, 5, 1])
            couleur = RISK_COLORS.get(a["niveau"], "#6c757d")
            c1.markdown(f"**{a['machine_id']}**<br><small>{a['type_machine']}</small>",
                        unsafe_allow_html=True)
            c2.markdown(f"<span style='color:{couleur};font-weight:bold'>{a['niveau'].upper()}</span>"
                        f"<br><small>{a['source']} — {a['created_at'][:19].replace('T', ' ')}</small>",
                        unsafe_allow_html=True)
            c3.write(a["message"])
            if c4.button("Acquitter", key=f"ack_{a['id']}"):
                _, err = api_post(f"/alertes/{a['id']}/acquitter")
                if err:
                    st.error(err)
                else:
                    st.rerun()

    st.markdown("---")

    df_recent  = pd.DataFrame(recent["predictions"])
    df_summary = pd.DataFrame(summary["machines"])

    if df_recent.empty:
        st.info(
            "Aucune donnée reçue pour le moment. "
            "Lancez le simulateur : `python src/sensor_simulator.py`"
        )
    else:
        niveau_max = regles.seuils_panne(REGLES)[0][0]
        niveau_2   = regles.seuils_panne(REGLES)[1][0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Prédictions (fenêtre)", len(df_recent))
        k2.metric(f"Risque {niveau_max}", int((df_recent["panne24h_risk_level"] == niveau_max).sum()))
        k3.metric(f"Risque {niveau_2}",   int((df_recent["panne24h_risk_level"] == niveau_2).sum()))
        k4.metric("Anomalies détectées",  int(df_recent["is_anomaly"].sum()))

        st.markdown("---")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.subheader("Répartition par niveau de risque")
            risk_counts = df_recent["panne24h_risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Niveau", "Nombre"]
            fig_risk = px.bar(
                risk_counts, x="Niveau", y="Nombre", color="Niveau",
                color_discrete_map=RISK_COLORS, title="Prédictions panne 24h par niveau",
            )
            st.plotly_chart(fig_risk, width="stretch")

        with col_v2:
            st.subheader("Anomalies détectées")
            ano_counts = df_recent["anomaly_risk_level"].value_counts().reset_index()
            ano_counts.columns = ["Niveau", "Nombre"]
            fig_ano = px.pie(
                ano_counts, names="Niveau", values="Nombre", color="Niveau",
                color_discrete_map=RISK_COLORS, title="Répartition Isolation Forest",
            )
            st.plotly_chart(fig_ano, width="stretch")

        st.markdown("---")

        if not df_summary.empty:
            st.subheader("État en direct par machine")
            usines_dispo = sorted(df_summary["usine_id"].unique())
            choix = st.multiselect("Usines", usines_dispo, default=usines_dispo)
            machines = df_summary[df_summary["usine_id"].isin(choix)].to_dict("records")
            for i in range(0, len(machines), 4):
                row_machines = machines[i:i + 4]
                cols = st.columns(len(row_machines))
                for col, m in zip(cols, row_machines):
                    with col:
                        mid = m["machine_id"].split("-M")[-1]
                        st.plotly_chart(
                            jauge(float(m.get("panne24h_probability", 0)),
                                  f"{m['usine_id']} M{mid}<br><sub>{m['type_machine']}</sub>",
                                  hauteur=200),
                            width="stretch", key=f"gauge_{m['machine_id']}",
                        )

        st.markdown("---")

        st.subheader("Dernières prédictions")
        display_cols = [
            "predicted_at", "machine_id", "usine_id", "type_machine",
            "panne24h_probability", "panne24h_risk_level",
            "rul_hours", "rul_risk_level",
            "is_anomaly", "anomaly_risk_level",
        ]
        df_disp = df_recent[[c for c in display_cols if c in df_recent.columns]].copy()
        df_disp["panne24h_probability"] = (df_disp["panne24h_probability"] * 100).round(1).astype(str) + " %"
        st.dataframe(df_disp, width="stretch", hide_index=True)

    if auto_refresh:
        time.sleep(rafraichissement)
        st.rerun()
