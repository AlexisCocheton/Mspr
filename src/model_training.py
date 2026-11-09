"""
Module d'entraînement des modèles ML pour MECHA.

Entraîne sur le dataset unifié mecha_unified_prepared.csv.

Tâches :
  1. Classification binaire : en_panne          (machine actuellement en panne)
  2. Classification binaire : panne_dans_24h    (panne imminente, usage prédictif)
  3. Régression             : rul_heures        (Remaining Useful Life)
  4. Détection d'anomalies  : Isolation Forest  (comportements atypiques)
  5. Validation de la règle d'alerte : modèle entraîné sur le passé, rejoué
     heure par heure sur la période la plus récente (comptage par événement)

Algorithmes testés : Random Forest, Gradient Boosting, Decision Tree,
                     Logistic Regression, KNN, SVM, Isolation Forest
Référence          : modèle naïf (répond toujours « pas de panne »)

Double évaluation des classifieurs :
  - Test rééquilibré 50/50 : vérifie que le modèle sait réellement reconnaître
    une panne (un modèle naïf y plafonne à 50 % d'accuracy, contre ~97 % sur
    les données réelles où les pannes sont rares). Sert à comparer les modèles.
  - Test sur la distribution réelle (~2-3 % de pannes, non rééquilibré) :
    mesure ce que verra la maintenance en production (précision des alertes,
    taux de fausses alertes par niveau de risque).

Régression RUL : découpage TEMPOREL (80 % les plus anciens / 20 % les plus
récents). Un découpage aléatoire place des heures voisines d'une même machine
dans le train et le test, ce qui surestime fortement la performance.
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.utils import resample

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

if str(BASE_DIR) not in sys.path:          # exécution directe : python src/model_training.py
    sys.path.insert(0, str(BASE_DIR))
from src import regles  # noqa: E402

PART_TEST = 0.2   # part du jeu réservée au test (aléatoire stratifié ou période la plus récente)

# Features utilisées pour l'entraînement
FEATURE_COLS = [
    # Capteurs instantanés (valeurs courantes du relevé)
    "temperature_C", "vibration_mm_s", "courant_A",
    "pression_bar",  "vitesse_tr_min",
    # Contexte machine
    "age_machine_h", "h_depuis_maintenance",
    # Catégorielles encodées
    "type_machine_encoded", "usine_encoded",
    # NOTE : les features glissantes 24h sont exclues volontairement.
    # Elles créent un signal temporel parfait (la moyenne intègre déjà l'état
    # actuel), rendant la classification triviale (F1≈1.0).
    # En production, seule la mesure courante est disponible en temps réel.
]


def _niveaux_alerte() -> list[tuple[str, float, float]]:
    """
    Niveaux de risque de config/regles_metier.yaml (les mêmes que l'API) sous la
    forme (niveau, probabilité min incluse, probabilité max exclue).
    Le dernier niveau (seuil 0, « faible ») n'émet pas d'alerte et est exclu.
    """
    niveaux, haut = [], np.inf
    for niveau, seuil in regles.seuils_panne():
        if seuil == 0:
            break
        niveaux.append((niveau, seuil, haut))
        haut = seuil
    return niveaux


NIVEAUX_ALERTE   = _niveaux_alerte()
SEUIL_ALERTE_MIN = NIVEAUX_ALERTE[-1][1]


def hyperparametres(model) -> dict:
    """Paramètres retenus du modèle (valeurs simples uniquement), pour la documentation."""
    return {k: v for k, v in model.get_params().items()
            if v is None or isinstance(v, (bool, int, float, str))}


def random_forest_classifier() -> RandomForestClassifier:
    """Classifieur déployé par l'API (panne 24 h et en_panne)."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "mecha_unified_prepared.csv"
    if not path.exists():
        raise FileNotFoundError("Lancez d'abord : python src/data_preparation.py")
    return pd.read_csv(path, parse_dates=["timestamp"])


def prepare_features(df: pd.DataFrame, target_col: str):
    """Extrait X et y en supprimant les lignes avec NaN dans les features."""
    df = df.dropna(subset=FEATURE_COLS).copy()
    X = df[FEATURE_COLS].fillna(0).values
    y = df[target_col].values
    return X, y


def balance_dataset(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Rééquilibre le dataset à 50/50 par sous-échantillonnage de la classe majoritaire.
    Permet d'évaluer les modèles sans biais lié au déséquilibre des classes.
    """
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos   = len(idx_pos)

    idx_neg_down = resample(idx_neg, n_samples=n_pos, replace=False, random_state=random_state)
    idx_balanced = np.concatenate([idx_pos, idx_neg_down])
    np.random.RandomState(random_state).shuffle(idx_balanced)

    return X[idx_balanced], y[idx_balanced]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Métriques de classification binaire (classe positive = panne)."""
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(auc),
    }


def alert_breakdown(y_true, y_proba) -> dict:
    """
    Répartit les prédictions par niveau d'alerte (seuils de l'API) et mesure,
    pour chaque niveau, la part d'alertes qui ne correspondent pas à une panne.
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    niveaux = {}
    for niveau, bas, haut in NIVEAUX_ALERTE:
        mask   = (y_proba >= bas) & (y_proba < haut)
        n      = int(mask.sum())
        vraies = int(y_true[mask].sum())
        niveaux[niveau] = {
            "n_alertes":            n,
            "vraies_pannes":        vraies,
            "taux_fausses_alertes": float(1 - vraies / n) if n else None,
        }
    non_alertees = int(((y_proba < SEUIL_ALERTE_MIN) & (y_true == 1)).sum())
    return {"niveaux": niveaux, "pannes_non_alertees": non_alertees}


def evaluate_real_distribution(y_true, y_pred, y_proba) -> dict:
    """Métriques sur le test NON rééquilibré (taux de pannes réel)."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["taux_pannes_test"] = float(np.mean(y_true))
    metrics["alertes"] = alert_breakdown(y_true, y_proba)
    return metrics


def evaluate_classifier(y_true, y_pred, y_proba, name: str, tag: str) -> dict:
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"  Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1_score']:.4f}  "
          f"AUC={metrics['auc_roc']:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion — {name} ({tag})")
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")
    fig.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}_{tag}.png"
    fig.savefig(MODELS_DIR / fname, dpi=150)
    plt.close(fig)

    return metrics


def plot_precision_recall(probas: dict, y_true, tag: str, modele_deploye: str = "random_forest"):
    """
    Courbes précision-rappel sur la distribution réelle.
    Les seuils d'alerte de l'API sont marqués sur la courbe du modèle déployé.
    """
    y_true = np.asarray(y_true)
    taux   = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(rec, prec, label=name, lw=2 if name == modele_deploye else 1)

    if modele_deploye in probas:
        p = np.asarray(probas[modele_deploye])
        for niveau, seuil, _ in NIVEAUX_ALERTE:
            alerte = p >= seuil
            if alerte.sum() == 0:
                continue
            rec_s  = y_true[alerte].sum() / y_true.sum()
            prec_s = y_true[alerte].mean()
            ax.scatter(rec_s, prec_s, color="black", zorder=5)
            ax.annotate(f"seuil {niveau} ({seuil:.0%})", (rec_s, prec_s),
                        textcoords="offset points", xytext=(-10, 8), ha="right", fontsize=8)

    ax.axhline(taux, color="grey", ls="--", label=f"Hasard (= taux de pannes {taux:.1%})")
    ax.set_xlabel("Rappel (part des pannes détectées)")
    ax.set_ylabel("Précision (part des alertes justifiées)")
    ax.set_title(f"Précision-rappel — distribution réelle ({tag})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"precision_recall_reel_{tag}.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, tag: str):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(FEATURE_COLS)), imp[idx][::-1], color="steelblue")
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {tag}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"feature_importance_{tag}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entraînement classification
# ---------------------------------------------------------------------------

def train_classification(X, y, tag: str) -> dict:
    """
    Entraîne 6 classifieurs + un modèle naïf de référence, sauvegarde les
    modèles + scaler. Chaque modèle est évalué sur le test 50/50 et sur le
    test à distribution réelle (clé « distribution_reelle »).
    """
    # Split avant rééquilibrage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PART_TEST, random_state=42, stratify=y
    )
    # Copie du test AVANT rééquilibrage : distribution réelle des pannes
    X_test_reel, y_test_reel = X_test, y_test
    # Rééquilibrage 50/50 sur train ET test
    X_train, y_train = balance_dataset(X_train, y_train)
    X_test,  y_test  = balance_dataset(X_test,  y_test)
    print(f"  Train : {len(y_train):,} obs (50/50) | Test : {len(y_test):,} obs (50/50)")
    print(f"  Test réel : {len(y_test_reel):,} obs ({y_test_reel.mean():.2%} de pannes)")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    Xte_reel = scaler.transform(X_test_reel)

    classifiers = {
        # Référence : répond toujours « pas de panne ». Montre pourquoi
        # l'accuracy sur données réelles ne suffit pas (~97 % sans rien détecter).
        "modele_naif": DummyClassifier(strategy="constant", constant=0),
        "random_forest": random_forest_classifier(),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "svm": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }

    results = {}
    trained = {}
    probas_reelles = {}

    for name, clf in classifiers.items():
        print(f"\n--- {name} ({tag}) ---")
        if name == "svm" and len(Xtr) > 10000:
            print("  (sous-échantillonnage SVM à 10 000)")
            idx = np.random.RandomState(42).choice(len(Xtr), 10000, replace=False)
            clf.fit(Xtr[idx], y_train[idx])
        else:
            clf.fit(Xtr, y_train)

        y_pred  = clf.predict(Xte)
        y_proba = clf.predict_proba(Xte)[:, 1]
        results[name] = evaluate_classifier(y_test, y_pred, y_proba, name, tag)
        results[name]["hyperparametres"] = hyperparametres(clf)
        trained[name] = clf

        # Évaluation complémentaire sur la distribution réelle
        y_pred_reel  = clf.predict(Xte_reel)
        y_proba_reel = clf.predict_proba(Xte_reel)[:, 1]
        reel = evaluate_real_distribution(y_test_reel, y_pred_reel, y_proba_reel)
        results[name]["distribution_reelle"] = reel
        probas_reelles[name] = y_proba_reel
        fa_crit = reel["alertes"]["niveaux"]["critique"]["taux_fausses_alertes"]
        print(f"  [réel] Accuracy={reel['accuracy']:.4f}  Precision={reel['precision']:.4f}  "
              f"Recall={reel['recall']:.4f}  F1={reel['f1_score']:.4f}  "
              f"Fausses alertes critiques="
              + (f"{fa_crit:.1%}" if fa_crit is not None else "aucune alerte"))

    # Validation croisée (RF, GB, DT)
    for name in ["random_forest", "gradient_boosting", "decision_tree"]:
        cv = cross_val_score(trained[name], Xtr, y_train, cv=5, scoring="f1")
        results[name]["cv_f1_mean"] = float(cv.mean())
        results[name]["cv_f1_std"]  = float(cv.std())
        print(f"CV F1 {name} : {cv.mean():.4f} ± {cv.std():.4f}")

    # Visualisations
    plot_feature_importance(trained["random_forest"],    f"rf_{tag}")
    plot_feature_importance(trained["gradient_boosting"], f"gb_{tag}")
    plot_precision_recall(
        {k: v for k, v in probas_reelles.items() if k != "modele_naif"},
        y_test_reel, tag,
    )

    # Sauvegarde (le modèle naïf sert uniquement de référence)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, clf in trained.items():
        if name == "modele_naif":
            continue
        joblib.dump(clf, MODELS_DIR / f"{name}_classifier_{tag}.joblib")
    joblib.dump(scaler,       MODELS_DIR / f"scaler_{tag}.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / f"feature_names_{tag}.joblib")
    print(f"\nModèles sauvegardés : {MODELS_DIR} (tag={tag})")

    return results


# ---------------------------------------------------------------------------
# Entraînement régression (RUL)
# ---------------------------------------------------------------------------

def split_temporel(df: pd.DataFrame, part_test: float = PART_TEST) -> tuple[np.ndarray, pd.Timestamp]:
    """
    Masque « train » : relevés antérieurs à la date de coupure, choisie pour que
    les `part_test` relevés les plus récents forment le test.
    """
    coupure = df["timestamp"].quantile(1 - part_test)
    return (df["timestamp"] < coupure).to_numpy(), coupure


def train_regression(X_train, X_test, y_train, y_test, tag: str) -> dict:
    """
    Entraîne RF et GB pour prédire les heures restantes avant panne, comparés à
    deux références : régression linéaire et modèle naïf (RUL moyen du train).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    regressors = {
        # Références de comparaison
        "modele_naif":         DummyRegressor(strategy="mean"),
        "regression_lineaire": LinearRegression(),
        # Modèles candidats
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42,
        ),
    }

    results = {}
    for name, reg in regressors.items():
        print(f"\n--- {name} RUL ({tag}) ---")
        reg.fit(Xtr, y_train)
        y_pred = reg.predict(Xte)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"  MAE={mae:.2f}h  RMSE={rmse:.2f}h  R²={r2:.4f}")
        results[name] = {
            "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
            "hyperparametres": hyperparametres(reg),
        }

        if name == "modele_naif":
            continue
        joblib.dump(reg, MODELS_DIR / f"{name}_regressor_{tag}.joblib")

        # Plot prédiction vs réel
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.2, s=8)
        lim = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([0, lim], [0, lim], "r--", label="Idéal")
        ax.set_xlabel("RUL réel (h)")
        ax.set_ylabel("RUL prédit (h)")
        ax.set_title(f"RUL — {name} ({tag})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(MODELS_DIR / f"rul_{name}_{tag}.png", dpi=150)
        plt.close(fig)

    joblib.dump(scaler, MODELS_DIR / f"scaler_rul_{tag}.joblib")
    return results


# ---------------------------------------------------------------------------
# Validation fonctionnelle de la règle d'alerte
# ---------------------------------------------------------------------------

HORIZON_H = 24               # une alerte est justifiée si la panne commence dans les 24 h
HORIZONS_PRECISION_H = (24, 72, 168)   # précision des alertes mesurée aussi à 3 et 7 jours
DELAI_ENTRE_ALERTES_H = 24   # approximation hors ligne de « alerte ouverte jusqu'à acquittement »


def debuts_de_panne(en_panne: np.ndarray) -> np.ndarray:
    """Indices où une panne commence (passage de 0 à 1)."""
    en_panne = np.asarray(en_panne).astype(int)
    precedent = np.concatenate([[0], en_panne[:-1]])
    return np.where((en_panne == 1) & (precedent == 0))[0]


def simuler_alertes(proba: np.ndarray, en_panne: np.ndarray, heures: np.ndarray,
                    seuil: float, lectures: int,
                    delai_h: float = DELAI_ENTRE_ALERTES_H) -> np.ndarray:
    """
    Rejoue la règle d'alerte sur la série horaire d'UNE machine.
    Alerte quand les `lectures` dernières probabilités sont >= seuil, en ignorant
    les heures où la machine est déjà en panne (arrêt connu du MES), puis
    aucune nouvelle alerte pendant `delai_h` heures.
    Retourne les heures (float) des alertes émises.
    """
    haut = (np.asarray(proba) >= seuil) & (np.asarray(en_panne) == 0)
    alertes, suite, derniere = [], 0, -np.inf
    for h, est_haut in zip(heures, haut):
        suite = suite + 1 if est_haut else 0
        if suite >= lectures and h - derniere >= delai_h:
            alertes.append(h)
            derniere = h
    return np.asarray(alertes, dtype=float)


def evaluer_alertes_evenements(alertes: np.ndarray, pannes: np.ndarray,
                               horizon_h: float = HORIZON_H) -> dict:
    """
    Compte par événement : une alerte est justifiée si une panne commence dans
    les `horizon_h` heures qui suivent ; une panne est anticipée si une alerte
    a été émise dans les `horizon_h` heures qui précèdent son début.
    """
    justifiees = sum(bool(np.any((pannes > a) & (pannes <= a + horizon_h))) for a in alertes)
    anticipations = []
    for p in pannes:
        avant = alertes[(alertes >= p - horizon_h) & (alertes < p)]
        if len(avant):
            anticipations.append(float(p - avant.min()))
    return {"n_alertes": len(alertes), "alertes_justifiees": justifiees,
            "n_pannes": len(pannes), "pannes_anticipees": len(anticipations),
            "anticipations_h": anticipations}


def alertes_suivies_de_panne(alertes: np.ndarray, pannes: np.ndarray,
                             fin_h: float, horizon_h: float) -> tuple[int, int]:
    """
    (alertes jugeables, alertes suivies d'un début de panne sous `horizon_h`).
    Seules les alertes émises au moins `horizon_h` heures avant la fin de la
    période observée (`fin_h`) sont jugeables.
    """
    jugeables = alertes[alertes <= fin_h - horizon_h]
    suivies = sum(bool(np.any((pannes > a) & (pannes <= a + horizon_h))) for a in jugeables)
    return len(jugeables), suivies


def valider_regle_alerte(df: pd.DataFrame) -> dict:
    """
    Validation en conditions proches de la production :
      - modèle panne 24 h (Random Forest, mêmes paramètres, train 50/50) entraîné
        sur les 80 % les plus anciens ;
      - rejoué heure par heure, machine par machine, sur les 20 % les plus récents ;
      - plusieurs règles candidates (seuil × lectures consécutives) comparées.
    Seules les pannes précédées d'au moins 24 h de test sont comptées.
    """
    df = df.dropna(subset=FEATURE_COLS).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    est_train, coupure = split_temporel(df)
    X, y = df[FEATURE_COLS].to_numpy(), df["panne_dans_24h"].to_numpy()

    Xb, yb = balance_dataset(X[est_train], y[est_train])
    scaler = StandardScaler().fit(Xb)
    clf = random_forest_classifier().fit(scaler.transform(Xb), yb)

    test = df[~est_train].copy()
    test["proba"] = clf.predict_proba(scaler.transform(X[~est_train]))[:, 1]
    test["heure"] = (test["timestamp"] - test["timestamp"].min()).dt.total_seconds() / 3600
    debut_eval = HORIZON_H

    series = []
    for _, g in test.groupby("machine_id", sort=False):
        heures = g["heure"].to_numpy()
        pannes = heures[debuts_de_panne(g["en_panne"].to_numpy())]
        series.append((g["proba"].to_numpy(), g["en_panne"].to_numpy(), heures,
                       pannes[pannes >= debut_eval]))
    machine_semaines = sum(len(s[2]) for s in series) / 168

    regle = regles.charger_regles()["alertes"]["panne24h"]
    seuils = dict(regles.seuils_panne())
    seuil_config = min(seuils[n] for n in regle["niveaux"])
    candidats = sorted({(s, n) for s in (0.5, 0.8) for n in (1, 2, 3, 4, 6)}
                       | {(seuil_config, int(regle["lectures_consecutives"]))})

    lignes = []
    for seuil, lectures in candidats:
        total = {"n_alertes": 0, "alertes_justifiees": 0, "n_pannes": 0,
                 "pannes_anticipees": 0, "anticipations_h": []}
        par_horizon = {h: [0, 0] for h in HORIZONS_PRECISION_H}
        for proba, en_panne, heures, pannes in series:
            alertes = simuler_alertes(proba, en_panne, heures, seuil, lectures)
            alertes = alertes[alertes >= debut_eval]
            for k, v in evaluer_alertes_evenements(alertes, pannes).items():
                total[k] += v
            for h in HORIZONS_PRECISION_H:
                jugeables, suivies = alertes_suivies_de_panne(alertes, pannes, heures.max(), h)
                par_horizon[h][0] += jugeables
                par_horizon[h][1] += suivies
        n_al, n_pa = total["n_alertes"], total["n_pannes"]
        precision_horizon = {f"{h}h": (s / j if j else None) for h, (j, s) in par_horizon.items()}
        lignes.append({
            "seuil": seuil,
            "lectures_consecutives": lectures,
            "regle_configuree": (seuil, lectures) == (seuil_config, int(regle["lectures_consecutives"])),
            "n_alertes": n_al,
            "n_pannes": n_pa,
            "rappel_evenements": total["pannes_anticipees"] / n_pa if n_pa else None,
            "precision_alertes": total["alertes_justifiees"] / n_al if n_al else None,
            "taux_fausses_alertes": 1 - total["alertes_justifiees"] / n_al if n_al else None,
            "part_alertes_suivies_de_panne": precision_horizon,
            "alertes_par_machine_semaine": n_al / machine_semaines,
            "anticipation_mediane_h": float(np.median(total["anticipations_h"]))
                                      if total["anticipations_h"] else None,
        })
        l = lignes[-1]
        print(f"  seuil={seuil:.2f} lectures={lectures}  alertes={n_al:>5}  "
              f"pannes anticipées={l['rappel_evenements'] or 0:.1%}  "
              f"fausses alertes 24h={l['taux_fausses_alertes'] or 0:.1%}  "
              f"suivies de panne sous 7j={precision_horizon['168h'] or 0:.0%}  "
              f"alertes/machine/sem={l['alertes_par_machine_semaine']:.2f}"
              + ("   <- règle configurée" if l["regle_configuree"] else ""))

    return {
        "protocole": (
            f"RF panne 24 h entraîné sur les relevés antérieurs au {coupure:%Y-%m-%d} (train 50/50), "
            f"rejoué heure par heure sur la période suivante ({len(series)} machines, "
            f"{machine_semaines:.0f} machine-semaines). Alerte justifiée si une panne commence "
            f"dans les {HORIZON_H} h ; un délai de {DELAI_ENTRE_ALERTES_H} h sépare deux alertes "
            f"d'une même machine."
        ),
        "candidats": lignes,
    }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def train_all():
    """Entraîne tous les modèles sur le dataset unifié MECHA."""
    df = load_dataset()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── 1. Classification : en_panne ─────────────────────────────────────────
    print("=" * 60)
    print("CLASSIFICATION : en_panne (machine en défaillance)")
    print("=" * 60)
    X, y = prepare_features(df, "en_panne")
    print(f"Échantillons : {len(y):,} | Positifs : {y.sum():,} ({y.mean():.2%})")
    all_results["classification_en_panne"] = train_classification(X, y, "mecha")

    # ── 2. Classification : panne_dans_24h ───────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION : panne_dans_24h (prédiction préventive)")
    print("=" * 60)
    X2, y2 = prepare_features(df, "panne_dans_24h")
    print(f"Échantillons : {len(y2):,} | Positifs : {y2.sum():,} ({y2.mean():.2%})")
    all_results["classification_panne_24h"] = train_classification(X2, y2, "mecha_24h")

    # ── 3. Régression : rul_heures ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉGRESSION : rul_heures (Remaining Useful Life)")
    print("=" * 60)
    # Exclure les enregistrements en panne (non prédictifs : le RUL y pointe vers la panne suivante)
    df_rul = df[df["en_panne"] == 0].dropna(subset=FEATURE_COLS)
    est_train, coupure = split_temporel(df_rul)
    X3, y3 = prepare_features(df_rul, "rul_heures")
    print(f"Échantillons : {len(y3):,} | coupure temporelle : {coupure:%Y-%m-%d %H:%M} "
          f"(train {est_train.sum():,} / test {(~est_train).sum():,})")
    all_results["regression_rul"] = train_regression(
        X3[est_train], X3[~est_train], y3[est_train], y3[~est_train], "mecha"
    )

    # ── 4. Détection d'anomalies : Isolation Forest ───────────────────────────
    print("\n" + "=" * 60)
    print("DÉTECTION D'ANOMALIES : Isolation Forest")
    print("=" * 60)
    all_results["anomaly_detection"] = train_anomaly_detection(df)

    # ── 5. Validation fonctionnelle de la règle d'alerte ─────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION DE LA RÈGLE D'ALERTE (rejeu temporel, comptage par événement)")
    print("=" * 60)
    all_results["validation_regle_alerte"] = valider_regle_alerte(df)

    all_results["protocole"] = {
        "classification": (
            "Split aléatoire stratifié 80/20. Train rééquilibré 50/50 ; évaluation sur un test "
            "rééquilibré 50/50 (capacité à reconnaître une panne, comparaison des modèles) ET sur "
            "le test à distribution réelle (clé distribution_reelle : ce que verra la maintenance)."
        ),
        "regression_rul": (
            f"Split temporel : train avant le {coupure:%Y-%m-%d}, test après. "
            "Relevés en panne exclus. Références : modèle naïf (moyenne) et régression linéaire."
        ),
        "anomaly_detection": "Isolation Forest entraîné sur les relevés à l'état normal uniquement.",
    }

    # Sauvegarde des métriques
    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats sauvegardés : {results_path}")

    return all_results


# ---------------------------------------------------------------------------
# Détection d'anomalies (Isolation Forest)
# ---------------------------------------------------------------------------

def train_anomaly_detection(df: pd.DataFrame) -> dict:
    """
    Entraîne un Isolation Forest sur les données normales uniquement.
    L'objectif est de détecter des comportements capteurs inhabituels
    sans nécessiter de labels de panne (approche non supervisée).
    """
    # Entraînement uniquement sur les machines en état normal
    df_normal = df[df["etat_machine"] == "normal"].dropna(subset=FEATURE_COLS)
    X_normal = df_normal[FEATURE_COLS].values

    print(f"Entraînement sur {len(X_normal):,} observations normales")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # 5% d'anomalies supposées dans les données
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Évaluation : sur l'ensemble du dataset, les pannes doivent être
    # majoritairement détectées comme anomalies
    df_eval = df.dropna(subset=FEATURE_COLS).copy()
    X_eval  = scaler.transform(df_eval[FEATURE_COLS].values)
    preds   = iso.predict(X_eval)          # -1 = anomalie, +1 = normal
    scores  = iso.score_samples(X_eval)    # plus bas = plus anormal

    df_eval["anomalie"] = (preds == -1).astype(int)

    # Taux de détection sur les pannes réelles
    mask_panne  = df_eval["en_panne"] == 1
    recall_panne = df_eval.loc[mask_panne, "anomalie"].mean()

    # Taux de faux positifs sur les normaux
    mask_normal = df_eval["etat_machine"] == "normal"
    fpr_normal  = df_eval.loc[mask_normal, "anomalie"].mean()

    print(f"  Recall sur pannes réelles     : {recall_panne:.2%}")
    print(f"  Faux positifs (état normal)   : {fpr_normal:.2%}")

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso,    MODELS_DIR / "isolation_forest_mecha.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_anomaly_mecha.joblib")
    print(f"  Modèle sauvegardé : isolation_forest_mecha.joblib")

    return {
        "recall_sur_pannes": float(recall_panne),
        "faux_positifs_normaux": float(fpr_normal),
        "n_train": int(len(X_normal)),
        "contamination": 0.05,
    }


if __name__ == "__main__":
    train_all()
