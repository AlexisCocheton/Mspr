"""
Dashboard Streamlit pour la maintenance prédictive MECHA.

Fournit aux équipes métiers (maintenance, production, pilotage) :
- Vue globale de l'état des machines et des usines
- Prédictions de pannes et indicateurs de risque
- Historique des données capteurs par machine
- Tableaux de bord de performance des modèles
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

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
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    return None


@st.cache_data
def load_training_results():
    path = MODELS_DIR / "training_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_category_maps():
    path = PROCESSED_DIR / "category_maps.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"type_machine": {}, "usine_id": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def risk_label(p: float) -> str:
    if p >= 0.8: return "CRITIQUE"
    if p >= 0.5: return "ELEVÉ"
    if p >= 0.3: return "MOYEN"
    return "FAIBLE"

RISK_COLORS = {
    "FAIBLE":  "#28a745",
    "MOYEN":   "#ffc107",
    "ELEVÉ":   "#fd7e14",
    "CRITIQUE":"#dc3545",
}

RISK_ACTIONS = {
    "CRITIQUE": "Arrêt immédiat recommandé",
    "ELEVÉ":    "Maintenance sous 24h",
    "MOYEN":    "Surveillance renforcée",
    "FAIBLE":   "Aucune action",
}

NUMERIC_FEATURES = [
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar", "vitesse_tr_min",
    "age_machine_h", "h_depuis_maintenance",
]

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
    ],
)


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

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Usines", df["usine_id"].nunique())
    with col2:
        st.metric("Machines", df["machine_id"].nunique())
    with col3:
        st.metric("Enregistrements", f"{len(df):,}")
    with col4:
        st.metric("Taux en panne", f"{df['en_panne'].mean():.2%}")
    with col5:
        if results and "classification_panne_24h" in results:
            rf = results["classification_panne_24h"].get("random_forest", {})
            f1 = rf.get("f1_score")
            if f1 is not None:
                st.metric("F1 (panne 24h)", f"{f1:.2%}")

    st.markdown("---")

    # Distribution des états par usine
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
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribution globale des états")
        etat_dist = df["etat_machine"].value_counts().reset_index()
        etat_dist.columns = ["État", "Count"]
        fig2 = px.pie(
            etat_dist, names="État", values="Count",
            title="Répartition des états machines",
            color="État", color_discrete_map=ETAT_COLORS,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Top 10 machines avec le plus de pannes
    st.subheader("Top 10 machines — nombre de pannes")
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
    st.plotly_chart(fig3, use_container_width=True)

    # Évolution temporelle
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
    st.plotly_chart(fig4, use_container_width=True)


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
            age          = st.slider("Âge machine (h)",        0, 50000, 10000, 100)

        with col3:
            h_maint      = st.slider("H depuis maintenance", 0, 5000, 300, 10)
            type_machine = st.selectbox("Type machine", types_machine)
            usine_id     = st.selectbox("Usine", usines)

        if st.button("Prédire", type="primary"):
            clf_path    = MODELS_DIR / "random_forest_classifier_mecha_24h.joblib"
            scaler_path = MODELS_DIR / "scaler_mecha_24h.joblib"
            rul_path    = MODELS_DIR / "random_forest_regressor_mecha.joblib"
            rul_sc_path = MODELS_DIR / "scaler_rul_mecha.joblib"

            if not clf_path.exists():
                st.error("Modèles non trouvés. Lancez d'abord `python src/model_training.py`.")
            else:
                clf     = joblib.load(clf_path)
                scaler  = joblib.load(scaler_path)
                type_enc  = maps["type_machine"].get(type_machine, 0)
                usine_enc = maps["usine_id"].get(usine_id, 0)

                features = np.array([[
                    temperature, vibration, courant, pression, vitesse,
                    age, h_maint,
                    type_enc, usine_enc,
                ]])

                features_scaled = scaler.transform(features)
                pred  = int(clf.predict(features_scaled)[0])
                proba = float(clf.predict_proba(features_scaled)[0][1])

                if proba >= 0.8:
                    risk, color = "CRITIQUE", "red"
                elif proba >= 0.5:
                    risk, color = "ELEVÉ",    "orange"
                elif proba >= 0.3:
                    risk, color = "MOYEN",    "yellow"
                else:
                    risk, color = "FAIBLE",   "green"

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Panne dans 24h", "OUI" if pred == 1 else "NON")
                with c2:
                    st.metric("Probabilité", f"{proba:.1%}")
                with c3:
                    st.markdown(f"**Risque :** :{color}[{risk}]")

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Probabilité de panne dans 24h (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "darkblue"},
                        "steps": [
                            {"range": [0,  30], "color": "#28a745"},
                            {"range": [30, 50], "color": "#ffc107"},
                            {"range": [50, 80], "color": "#fd7e14"},
                            {"range": [80,100], "color": "#dc3545"},
                        ],
                    },
                ))
                st.plotly_chart(fig, use_container_width=True)

                # RUL
                if rul_path.exists():
                    reg      = joblib.load(rul_path)
                    rul_sc   = joblib.load(rul_sc_path)
                    rul_feat = rul_sc.transform(features)
                    rul = max(0, float(reg.predict(rul_feat)[0]))
                    st.metric("RUL estimé", f"{rul:.0f} heures")

    # -----------------------------------------------------------------------
    # Onglet 2 : prédiction par CSV
    # -----------------------------------------------------------------------
    with tab_csv:
        st.subheader("Rapport de pannes potentielles via CSV")

        SAMPLE_COLS = ["machine_id", "type_machine", "usine_id"] + NUMERIC_FEATURES

        # --- CSV exemple téléchargeable (format brut lisible) ---
        st.markdown("#### Télécharger un CSV exemple")
        df_full = load_data()
        if df_full is not None:
            pannes  = df_full[df_full["en_panne"] == 1][SAMPLE_COLS + ["en_panne"]].sample(
                n=min(50, (df_full["en_panne"] == 1).sum()), random_state=42
            )
            normaux = df_full[df_full["en_panne"] == 0][SAMPLE_COLS + ["en_panne"]].sample(
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
                st.caption(f"Colonnes optionnelles : `machine_id, type_machine, usine_id`")
            with col_prev:
                with st.expander("Aperçu"):
                    st.dataframe(sample_csv.head(5), use_container_width=True)
        else:
            st.info("Données non disponibles pour générer le CSV exemple.")

        st.markdown("---")

        # --- Upload ---
        uploaded = st.file_uploader("Uploader un CSV pour générer le rapport", type=["csv"])

        if uploaded is not None:
            clf_path    = MODELS_DIR / "random_forest_classifier_mecha_24h.joblib"
            scaler_path = MODELS_DIR / "scaler_mecha_24h.joblib"
            rul_path    = MODELS_DIR / "random_forest_regressor_mecha.joblib"
            rul_sc_path = MODELS_DIR / "scaler_rul_mecha.joblib"

            if not clf_path.exists():
                st.error("Modèles non trouvés. Lancez d'abord `python src/model_training.py`.")
            else:
                try:
                    df_upload = pd.read_csv(uploaded)

                    # Vérification colonnes numériques obligatoires
                    missing = [c for c in NUMERIC_FEATURES if c not in df_upload.columns]
                    if missing:
                        st.error(f"Colonnes manquantes dans le CSV : {missing}")
                        st.stop()

                    maps = load_category_maps()

                    # Encodage des catégorielles (rétro-compatible)
                    if "type_machine_encoded" in df_upload.columns:
                        type_enc = df_upload["type_machine_encoded"].values
                    elif "type_machine" in df_upload.columns:
                        type_enc = df_upload["type_machine"].map(maps["type_machine"]).fillna(0).astype(int).values
                    else:
                        type_enc = np.zeros(len(df_upload), dtype=int)

                    if "usine_encoded" in df_upload.columns:
                        usine_enc = df_upload["usine_encoded"].values
                    elif "usine_id" in df_upload.columns:
                        usine_enc = df_upload["usine_id"].map(maps["usine_id"]).fillna(0).astype(int).values
                    else:
                        usine_enc = np.zeros(len(df_upload), dtype=int)

                    X_num = df_upload[NUMERIC_FEATURES].values
                    X = np.column_stack([X_num, type_enc, usine_enc])

                    clf    = joblib.load(clf_path)
                    scaler = joblib.load(scaler_path)
                    X_sc   = scaler.transform(X)
                    preds  = clf.predict(X_sc)
                    probas = clf.predict_proba(X_sc)[:, 1]

                    df_result = df_upload.copy()
                    # Ajouter machine_id si absente
                    if "machine_id" not in df_result.columns:
                        df_result.insert(0, "machine_id", [f"Machine-{i+1}" for i in range(len(df_result))])
                    df_result["panne_dans_24h"] = preds
                    df_result["probabilite"]    = probas.round(4)
                    df_result["risque"]         = [risk_label(p) for p in probas]
                    df_result["action"]         = df_result["risque"].map(RISK_ACTIONS)

                    # RUL
                    has_rul = rul_path.exists()
                    if has_rul:
                        reg    = joblib.load(rul_path)
                        rul_sc = joblib.load(rul_sc_path)
                        ruls   = reg.predict(rul_sc.transform(X))
                        df_result["rul_estime_h"] = np.maximum(0, ruls).round(1)

                    st.success(f"{len(df_result)} lignes analysées.")

                    # -----------------------------------------------------------
                    # A — Résumé
                    # -----------------------------------------------------------
                    nb_risque   = int((probas >= 0.3).sum())
                    nb_critique = int((probas >= 0.8).sum())
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Lignes analysées", len(df_result))
                    k2.metric("Machines à risque", nb_risque)
                    k3.metric("Dont critiques", nb_critique)
                    if has_rul and nb_risque > 0:
                        rul_moyen = df_result.loc[df_result["probabilite"] >= 0.3, "rul_estime_h"].mean()
                        k4.metric("RUL moyen (à risque)", f"{rul_moyen:.0f} h")

                    st.markdown("---")

                    # -----------------------------------------------------------
                    # B — Rapport des pannes potentielles
                    # -----------------------------------------------------------
                    st.subheader("Rapport des pannes potentielles")

                    df_risque = df_result[df_result["probabilite"] >= 0.3].sort_values(
                        "probabilite", ascending=False
                    ).reset_index(drop=True)

                    if df_risque.empty:
                        st.success("Aucune machine à risque détectée dans ce fichier.")
                    else:
                        # Colonnes affichées dans le rapport
                        rapport_cols = ["machine_id"]
                        if "type_machine" in df_risque.columns:
                            rapport_cols.append("type_machine")
                        if "usine_id" in df_risque.columns:
                            rapport_cols.append("usine_id")
                        rapport_cols += ["probabilite", "risque"]
                        if has_rul:
                            rapport_cols.append("rul_estime_h")
                        rapport_cols.append("action")

                        df_rapport = df_risque[rapport_cols].copy()
                        df_rapport["probabilite"] = (df_rapport["probabilite"] * 100).round(1).astype(str) + " %"

                        st.dataframe(
                            df_rapport,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            label=f"Télécharger le rapport ({len(df_risque)} machines à risque)",
                            data=df_risque[rapport_cols].to_csv(index=False).encode("utf-8"),
                            file_name="rapport_pannes_mecha.csv",
                            mime="text/csv",
                        )

                    st.markdown("---")

                    # -----------------------------------------------------------
                    # C — Visualisations
                    # -----------------------------------------------------------
                    col_viz1, col_viz2 = st.columns(2)

                    with col_viz1:
                        # Camembert répartition par niveau de risque
                        risk_counts = df_result["risque"].value_counts().reset_index()
                        risk_counts.columns = ["risque", "count"]
                        fig_pie = px.pie(
                            risk_counts, values="count", names="risque",
                            title="Répartition par niveau de risque",
                            color="risque",
                            color_discrete_map=RISK_COLORS,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_viz2:
                        # Barres RUL des machines à risque (plus urgent = RUL le plus faible)
                        if has_rul and not df_risque.empty:
                            df_rul_chart = df_risque.nsmallest(20, "rul_estime_h")[
                                ["machine_id", "rul_estime_h", "risque"]
                            ].sort_values("rul_estime_h")
                            fig_rul = px.bar(
                                df_rul_chart, x="machine_id", y="rul_estime_h",
                                color="risque",
                                color_discrete_map=RISK_COLORS,
                                title="RUL des 20 machines les plus urgentes (heures restantes)",
                                labels={"rul_estime_h": "RUL (h)", "machine_id": "Machine"},
                            )
                            fig_rul.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_rul, use_container_width=True)
                        elif not has_rul:
                            st.info("Modèle RUL non disponible.")

                    st.markdown("---")

                    # -----------------------------------------------------------
                    # D — Données complètes
                    # -----------------------------------------------------------
                    with st.expander("Données complètes (toutes les lignes)"):
                        st.dataframe(df_result, use_container_width=True)
                        st.download_button(
                            label="Télécharger le CSV complet avec prédictions",
                            data=df_result.to_csv(index=False).encode("utf-8"),
                            file_name="mecha_predictions_completes.csv",
                            mime="text/csv",
                        )

                    # -----------------------------------------------------------
                    # E — Comparaison labels réels (si présents)
                    # -----------------------------------------------------------
                    if "en_panne" in df_upload.columns or "panne_dans_24h" in df_upload.columns:
                        label_col = "panne_dans_24h" if "panne_dans_24h" in df_upload.columns else "en_panne"
                        y_true = df_upload[label_col].values
                        from sklearn.metrics import classification_report
                        report = classification_report(y_true, preds, output_dict=True)
                        st.subheader("Comparaison avec les labels réels")
                        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

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

    # Sélection machine
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

    # Courbes temporelles des capteurs
    st.subheader(f"Capteurs — {selected_machine} (500 dernières mesures)")
    tail = machine_df.tail(500)

    for col in SENSOR_COLS:
        fig = px.line(tail, x="timestamp", y=col, title=SENSOR_LABELS[col])
        # Zones de panne en rouge
        pannes = tail[tail["en_panne"] == 1]
        if not pannes.empty:
            fig.add_trace(go.Scatter(
                x=pannes["timestamp"], y=pannes[col],
                mode="markers",
                marker=dict(color="red", size=4),
                name="En panne",
            ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Statistiques et corrélations
    st.subheader("Corrélation capteurs × score_degradation")
    corr_cols = SENSOR_COLS + ["score_degradation", "rul_heures", "en_panne"]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", title="Matrice de corrélation",
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distribution des capteurs par état machine")
    selected_sensor = st.selectbox("Capteur", SENSOR_COLS,
                                   format_func=lambda x: SENSOR_LABELS[x])
    fig_dist = px.box(
        df, x="etat_machine", y=selected_sensor,
        color="etat_machine", color_discrete_map=ETAT_COLORS,
        title=f"{SENSOR_LABELS[selected_sensor]} par état machine",
        category_orders={"etat_machine": ["normal", "a_risque", "critique", "en_panne"]},
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 : Performance des modèles
# ---------------------------------------------------------------------------

elif page == "Performance des modèles":
    st.title("Performance des modèles de Machine Learning")

    results = load_training_results()
    if results is None:
        st.warning("Aucun résultat trouvé. Lancez d'abord `python src/model_training.py`.")
        st.stop()

    TASK_LABELS = {
        "classification_en_panne":    "Classification — en_panne",
        "classification_panne_24h":   "Classification — panne dans 24h",
        "regression_rul":             "Régression — RUL (heures)",
    }

    for task_key, task_results in results.items():
        label = TASK_LABELS.get(task_key, task_key)
        st.subheader(label)

        if not task_results:
            st.info("Aucun résultat disponible pour cette tâche.")
        elif "regression" in task_key:
            # Tableau de régression
            rows = [{"Modèle": k, **v} for k, v in task_results.items()
                    if isinstance(v, dict)]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            # Tableau de classification
            # anomaly_detection a une structure plate {metric: value}, pas {model: {metric: value}}
            first_val = next(iter(task_results.values()))
            if not isinstance(first_val, dict):
                st.dataframe(
                    pd.DataFrame([task_results]).rename(columns=lambda c: c.replace("_", " ").title()),
                    use_container_width=True,
                )
            else:
                rows = []
                for model_name, metrics in task_results.items():
                    row = {"Modèle": model_name.replace("_", " ").title()}
                    row.update({k: v for k, v in metrics.items()
                                 if k in ("accuracy", "precision", "recall", "f1_score", "auc_roc")})
                    rows.append(row)
                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df, use_container_width=True)

                # Graphique comparatif
                fig = go.Figure()
                for _, row in metrics_df.iterrows():
                    fig.add_trace(go.Bar(
                        name=row["Modèle"],
                        x=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
                        y=[row.get("accuracy", 0), row.get("precision", 0),
                           row.get("recall", 0),   row.get("f1_score", 0),
                           row.get("auc_roc", 0)],
                    ))
                fig.update_layout(
                    title=f"Comparaison des modèles — {label}",
                    barmode="group",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # Images sauvegardées
    st.subheader("Visualisations sauvegardées")
    imgs = sorted(MODELS_DIR.glob("*.png"))
    if imgs:
        for img in imgs:
            st.image(str(img), caption=img.stem.replace("_", " ").title())
    else:
        st.info("Aucune image. Lancez l'entraînement pour générer les graphiques.")
