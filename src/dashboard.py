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
    page_icon="🏭",
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
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🏭 MECHA")
st.sidebar.markdown("### Maintenance Prédictive IA")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Vue d'ensemble",
        "🔍 Prédiction en temps réel",
        "📈 Analyse des capteurs",
        "🤖 Performance des modèles",
    ],
)


# ---------------------------------------------------------------------------
# Page 1 : Vue d'ensemble
# ---------------------------------------------------------------------------

if page == "📊 Vue d'ensemble":
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
            f1 = results["classification_panne_24h"]["random_forest"]["f1_score"]
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

elif page == "🔍 Prédiction en temps réel":
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
                    risk, color = "ÉLEVÉ",    "orange"
                elif proba >= 0.3:
                    risk, color = "MOYEN",    "yellow"
                else:
                    risk, color = "FAIBLE",   "green"

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Panne dans 24h", "⚠️ OUI" if pred == 1 else "✅ NON")
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
                    rul_feat = scaler.transform(features) if scaler_path == rul_sc_path \
                               else rul_sc.transform(features)
                    rul = max(0, float(reg.predict(rul_feat)[0]))
                    st.metric("RUL estimé", f"{rul:.0f} heures")

    # -----------------------------------------------------------------------
    # Onglet 2 : prédiction par CSV
    # -----------------------------------------------------------------------
    with tab_csv:
        st.subheader("Prédictions en batch via CSV")

        FEATURE_COLS_CSV = [
            "temperature_C", "vibration_mm_s", "courant_A",
            "pression_bar", "vitesse_tr_min",
            "age_machine_h", "h_depuis_maintenance",
            "type_machine_encoded", "usine_encoded",
        ]

        # --- Téléchargement d'un CSV exemple 50/50 ---
        st.markdown("#### Télécharger un CSV exemple (50/50 panne / normal)")
        df_full = load_data()
        if df_full is not None:
            pannes  = df_full[df_full["en_panne"] == 1][FEATURE_COLS_CSV + ["en_panne"]].sample(
                n=min(50, (df_full["en_panne"] == 1).sum()), random_state=42
            )
            normaux = df_full[df_full["en_panne"] == 0][FEATURE_COLS_CSV + ["en_panne"]].sample(
                n=50, random_state=42
            )
            sample_csv = pd.concat([pannes, normaux]).sample(frac=1, random_state=42).reset_index(drop=True)
            st.download_button(
                label="Télécharger le CSV exemple (100 lignes, 50/50)",
                data=sample_csv.to_csv(index=False).encode("utf-8"),
                file_name="mecha_sample_50_50.csv",
                mime="text/csv",
            )
            with st.expander("Aperçu du CSV exemple"):
                st.dataframe(sample_csv.head(10), use_container_width=True)
        else:
            st.info("Données non disponibles pour générer le CSV exemple.")

        st.markdown("---")

        # --- Upload et prédiction ---
        st.markdown("#### Uploader un CSV pour prédictions en batch")
        st.caption(f"Colonnes attendues : `{', '.join(FEATURE_COLS_CSV)}`")

        uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])

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
                    missing = [c for c in FEATURE_COLS_CSV if c not in df_upload.columns]
                    if missing:
                        st.error(f"Colonnes manquantes dans le CSV : {missing}")
                    else:
                        clf    = joblib.load(clf_path)
                        scaler = joblib.load(scaler_path)
                        X      = df_upload[FEATURE_COLS_CSV].values
                        X_sc   = scaler.transform(X)
                        preds  = clf.predict(X_sc)
                        probas = clf.predict_proba(X_sc)[:, 1]

                        def risk_label(p):
                            if p >= 0.8: return "CRITIQUE"
                            if p >= 0.5: return "ÉLEVÉ"
                            if p >= 0.3: return "MOYEN"
                            return "FAIBLE"

                        df_result = df_upload.copy()
                        df_result["panne_dans_24h"] = preds
                        df_result["probabilite"]    = probas.round(4)
                        df_result["risque"]         = [risk_label(p) for p in probas]

                        # RUL si disponible
                        if rul_path.exists():
                            reg    = joblib.load(rul_path)
                            rul_sc = joblib.load(rul_sc_path)
                            ruls   = reg.predict(rul_sc.transform(X))
                            df_result["rul_estime_h"] = np.maximum(0, ruls).round(1)

                        st.success(f"{len(df_result)} lignes traitées.")

                        # KPIs
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Pannes prédites", int(preds.sum()))
                        k2.metric("Taux de panne", f"{preds.mean():.1%}")
                        k3.metric("Probabilité moyenne", f"{probas.mean():.1%}")

                        # Graphique distribution des probabilités
                        fig_hist = px.histogram(
                            df_result, x="probabilite", color="risque",
                            nbins=20,
                            title="Distribution des probabilités de panne",
                            color_discrete_map={
                                "FAIBLE": "#28a745", "MOYEN": "#ffc107",
                                "ÉLEVÉ": "#fd7e14",  "CRITIQUE": "#dc3545",
                            },
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Tableau résultat
                        st.subheader("Résultats détaillés")
                        st.dataframe(df_result, use_container_width=True)

                        # Téléchargement résultats
                        st.download_button(
                            label="Télécharger les résultats",
                            data=df_result.to_csv(index=False).encode("utf-8"),
                            file_name="mecha_predictions.csv",
                            mime="text/csv",
                        )

                        # Comparaison avec label réel si présent
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

elif page == "📈 Analyse des capteurs":
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

elif page == "🤖 Performance des modèles":
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
        st.subheader(f"📋 {label}")

        if "regression" in task_key:
            # Tableau de régression
            rows = [{"Modèle": k, **v} for k, v in task_results.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            # Tableau de classification
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
