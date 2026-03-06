"""
Module d'entraînement des modèles ML pour MECHA.

Entraîne sur le dataset unifié mecha_unified_prepared.csv.

Tâches :
  1. Classification binaire : en_panne          (machine actuellement en panne)
  2. Classification binaire : panne_dans_24h    (panne imminente, usage prédictif)
  3. Régression             : rul_heures        (Remaining Useful Life)
  4. Détection d'anomalies  : Isolation Forest  (comportements atypiques)

Algorithmes testés : Random Forest, Gradient Boosting, Decision Tree,
                     Logistic Regression, KNN, SVM, Isolation Forest
"""

import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.utils import resample

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

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


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "mecha_unified_prepared.csv"
    if not path.exists():
        raise FileNotFoundError("Lancez d'abord : python src/data_preparation.py")
    return pd.read_csv(path)


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


def evaluate_classifier(y_true, y_pred, y_proba, name: str, tag: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0

    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
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

    return {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1_score":  float(f1),
        "auc_roc":   float(auc),
    }


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
    """Entraîne 6 classifieurs, sauvegarde tous les modèles + scaler."""
    # Split avant rééquilibrage pour ne pas contaminer le test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Rééquilibrage 50/50 sur le train uniquement
    X_train, y_train = balance_dataset(X_train, y_train)
    print(f"  Train rééquilibré : {len(y_train):,} observations (50% panne / 50% normal)")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
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
        trained[name] = clf

    # Validation croisée (RF, GB, DT)
    for name in ["random_forest", "gradient_boosting", "decision_tree"]:
        cv = cross_val_score(trained[name], Xtr, y_train, cv=5, scoring="f1")
        results[name]["cv_f1_mean"] = float(cv.mean())
        results[name]["cv_f1_std"]  = float(cv.std())
        print(f"CV F1 {name} : {cv.mean():.4f} ± {cv.std():.4f}")

    # Visualisations
    plot_feature_importance(trained["random_forest"],    f"rf_{tag}")
    plot_feature_importance(trained["gradient_boosting"], f"gb_{tag}")

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, clf in trained.items():
        joblib.dump(clf, MODELS_DIR / f"{name}_classifier_{tag}.joblib")
    joblib.dump(scaler,       MODELS_DIR / f"scaler_{tag}.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / f"feature_names_{tag}.joblib")
    print(f"\nModèles sauvegardés : {MODELS_DIR} (tag={tag})")

    return results


# ---------------------------------------------------------------------------
# Entraînement régression (RUL)
# ---------------------------------------------------------------------------

def train_regression(X, y, tag: str) -> dict:
    """Entraîne RF et GB pour prédire les heures restantes avant panne."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    regressors = {
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
        results[name] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

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
    # Exclure les enregistrements en panne (RUL = 0, non prédictifs)
    df_rul = df[df["en_panne"] == 0].copy()
    X3, y3 = prepare_features(df_rul, "rul_heures")
    print(f"Échantillons : {len(y3):,}")
    all_results["regression_rul"] = train_regression(X3, y3, "mecha")

    # ── 4. Détection d'anomalies : Isolation Forest ───────────────────────────
    print("\n" + "=" * 60)
    print("DÉTECTION D'ANOMALIES : Isolation Forest")
    print("=" * 60)
    all_results["anomaly_detection"] = train_anomaly_detection(df)

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
