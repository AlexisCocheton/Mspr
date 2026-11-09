"""
MECHA Industrial AI Project
============================
Script de génération du jeu de données synthétique
Simule des capteurs IoT sur 5 usines (3 France, 2 Espagne)
Cas d'usage : Maintenance Prédictive

Auteurs : Équipe MSPR TPRE841
Date    : 2025-2026

Corrélations physiques implémentées :
  1. score_degradation ↑  → température ↑, vibration ↑, courant ↑,
                             pression ↓, vitesse ↓
  2. en_panne = 1         → valeurs capteurs critiques/anormales
                             (température surchauffe, vibration explosive,
                              pression effondrée, vitesse quasi-nulle)
  3. age_machine_h ↑      → MTBF réduit (pannes plus fréquentes)
  4. h_depuis_maintenance ↑ → dérive progressive des capteurs
                              (sans entretien → usure → valeurs hors-norme)
  5. Saisonnalité          → été (juin-août) en Espagne/sud +10 °C ambiants
                              → légère augmentation température machine

Hétérogénéité et qualité des données (fichier BRUT, avant nettoyage) :
  6. Sites espagnols équipés plus récemment → historique plus court (12 mois
     au lieu de 18) et capteurs plus sujets aux pertes de mesure.
  7. Défauts capteurs injectés après simulation (cf. injecter_defauts_qualite) :
     valeurs manquantes, pertes de transmission, capteurs figés, valeurs
     aberrantes, doublons de transmission. Les labels ne sont pas modifiés.
     Le nettoyage est fait par src/data_preparation.py.

Sortie : data/raw/mecha_dataset_full.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

# ─── Paramètres de simulation ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

USINES = {
    "USN-FR-01": {"pays": "France",  "ville": "Toulouse",  "nb_machines": 8,  "climat": "chaud"},
    "USN-FR-02": {"pays": "France",  "ville": "Lyon",      "nb_machines": 6,  "climat": "tempere"},
    "USN-FR-03": {"pays": "France",  "ville": "Bordeaux",  "nb_machines": 7,  "climat": "chaud"},
    "USN-ES-01": {"pays": "Espagne", "ville": "Barcelone", "nb_machines": 5,  "climat": "mediterraneen"},
    "USN-ES-02": {"pays": "Espagne", "ville": "Valence",   "nb_machines": 4,  "climat": "mediterraneen"},
}

TYPE_MACHINES = ["CNC-Fraisage", "CNC-Tournage", "Découpe-Laser", "Centre-Usinage"]

# Durée de simulation : 18 mois (granularité 1 heure)
DATE_DEBUT = datetime(2024, 1, 1)
DATE_FIN   = datetime(2025, 6, 30)

# Corrélation 6 : capteurs IoT déployés plus tard sur les sites espagnols
DEBUT_COLLECTE_PAYS = {"France": DATE_DEBUT, "Espagne": datetime(2024, 7, 1)}

SORTIE = Path(__file__).resolve().parent.parent / "data" / "raw" / "mecha_dataset_full.csv"

CAPTEURS = ["temperature_C", "vibration_mm_s", "courant_A", "pression_bar", "vitesse_tr_min"]

# Corrélation 7 : défauts de qualité injectés dans le fichier brut
DEFAUTS = {
    "taux_valeur_manquante":      {"France": 0.005, "Espagne": 0.012},  # par capteur et par relevé
    "pertes_transmission_par_an": 6,          # coupures réseau par machine (tous capteurs absents)
    "duree_perte_h":              (1, 6),
    "capteurs_figes_par_an":      3,          # épisodes par machine
    "duree_capteur_fige_h":       (6, 24),
    "taux_valeur_aberrante":      0.0005,     # par capteur et par relevé
    "taux_doublon":               0.001,      # relevés transmis deux fois
}
# Valeurs physiquement impossibles typiques d'un capteur en erreur
VALEURS_ABERRANTES = {
    "temperature_C":  [999.9, -273.0],
    "vibration_mm_s": [99.99, -1.0],
    "courant_A":      [-5.0, 500.0],
    "pression_bar":   [-1.0, 99.0],
    "vitesse_tr_min": [-1.0, 65535.0],
}

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def age_machine_aleatoire():
    """Retourne un âge machine en heures (entre 1 000 et 50 000h)."""
    return random.randint(1000, 50000)


def facteur_saisonnier(date, climat):
    """
    Retourne un delta de température ambiante selon la saison et le climat.
    Corrélation : chaleur estivale → machines plus chaudes → risque accru.
    """
    mois = date.month
    if mois in (6, 7, 8):      # été
        if climat == "mediterraneen":
            return np.random.normal(10, 2)   # +10 °C en moyenne
        elif climat == "chaud":
            return np.random.normal(6, 1.5)
        else:
            return np.random.normal(3, 1)
    elif mois in (12, 1, 2):   # hiver → machines se refroidissent plus vite, vibration légèrement réduite
        return np.random.normal(-3, 1)
    else:
        return np.random.normal(0, 0.5)


def simuler_degradation(heures_restantes_avant_panne, h_depuis_maintenance, bruit=True):
    """
    Calcule un score de dégradation [0, 1].
    Score proche de 1 → dégradation avancée (panne imminente).

    Corrélation 4 : h_depuis_maintenance contribue à la dérive du score
    → plus la machine est longtemps sans entretien, plus elle dégrade vite.
    """
    # Composante principale : proximité de la prochaine panne
    score_rul = 1 - (heures_restantes_avant_panne / 720)
    score_rul = np.clip(score_rul, 0, 1)

    # Composante maintenance : dérive progressive sans entretien
    # Plafonné à 0.25 pour ne pas dominer le signal RUL
    score_maint = np.clip(h_depuis_maintenance / 4000, 0, 0.25)

    score = np.clip(score_rul + score_maint * (1 - score_rul), 0, 1)

    if bruit:
        score += np.random.normal(0, 0.03)
        score = np.clip(score, 0, 1)

    return round(score, 4)


def simuler_capteurs(score_degradation, type_machine, date, climat,
                     h_depuis_maintenance, mode_critique=False, en_panne=False):
    """
    Génère des valeurs capteurs cohérentes avec le niveau de dégradation.

    Corrélations implémentées :
      • score_degradation ↑  → temp ↑, vibration ↑, courant ↑, pression ↓, vitesse ↓
      • en_panne = True      → valeurs critiques (surchauffe, vibration explosive, pression effondrée)
      • h_depuis_maintenance ↑ → dérive additionnelle (lubrification insuffisante → frottement)
      • Saisonnalité          → température ambiante affecte la température machine
    """
    # Valeurs nominales par type de machine (état sain, bien entretenu)
    config = {
        "CNC-Fraisage":   {"temp_base": 65, "vib_base": 2.1, "courant_base": 18},
        "CNC-Tournage":   {"temp_base": 60, "vib_base": 1.8, "courant_base": 15},
        "Découpe-Laser":  {"temp_base": 45, "vib_base": 0.8, "courant_base": 22},
        "Centre-Usinage": {"temp_base": 70, "vib_base": 2.5, "courant_base": 25},
    }
    cfg = config[type_machine]

    # ── Corrélation 5 : saisonnalité ──────────────────────────────────────────
    delta_temp_ambiant = facteur_saisonnier(date, climat)

    # ── Corrélation 4 : dérive due au manque de maintenance ───────────────────
    # Manque de lubrification → frottement → chaleur et vibration supplémentaires
    drift_maint = np.clip(h_depuis_maintenance / 2000, 0, 0.30)

    if en_panne:
        # ── Corrélation 2 : machine EN PANNE ─────────────────────────────────
        # Comportement erratique : signal fort mais bruité, chevauchement
        # intentionnel avec l'état "critique" pour un F1 cible réaliste.
        score_eff  = min(1.0, score_degradation + np.random.uniform(0.0, 0.12))
        drift_eff  = min(0.50, drift_maint       + np.random.uniform(0.0, 0.18))
        facteur    = 1 + score_eff * 0.30 + drift_eff   # amplitude réduite : 0.45→0.30

        temperature_C  = cfg["temp_base"] * facteur + delta_temp_ambiant + np.random.normal(0, 16)
        vibration_mm_s = (cfg["vib_base"] * facteur * np.random.uniform(1.05, 1.20)
                          + np.random.normal(0, 0.9))
        courant_A      = cfg["courant_base"] * facteur + np.random.normal(0, 7.0)
        pression_bar   = round(max(1.0, 6.5 - score_eff * 1.10 - drift_eff * 0.55
                                   + np.random.normal(0, 0.75)), 2)
        vitesse_tr_min = round(max(0, 1500 - score_eff * 380 - drift_eff * 140
                                   + np.random.normal(0, 140)), 0)

    else:
        # ── Corrélations 1 + 4 : dégradation progressive ─────────────────────
        facteur = 1 + score_degradation * 0.30 + drift_maint   # amplitude réduite : 0.45→0.30

        if mode_critique:
            facteur *= 1.07  # surchauffe modérée dans les 48h avant panne

        # Bruit industriel réaliste : interférences, vibrations parasites, etc.
        noise_factor = 1 + np.random.normal(0, 0.04)   # ±4 % bruit multiplicatif

        temperature_C  = cfg["temp_base"] * facteur * noise_factor + delta_temp_ambiant + np.random.normal(0, 10)
        vibration_mm_s = cfg["vib_base"]  * facteur * noise_factor + drift_maint * 0.5 + np.random.normal(0, 0.55)
        courant_A      = cfg["courant_base"] * facteur * noise_factor + np.random.normal(0, 3.5)
        pression_bar   = round(6.5 - score_degradation * 0.9 - drift_maint * 0.4
                               + np.random.normal(0, 0.45), 2)
        vitesse_tr_min = round(max(100, 1500 - score_degradation * 300
                                   - drift_maint * 80 + np.random.normal(0, 90)), 0)

        # Pics capteurs aléatoires (1% de chance) : bruit industriel ponctuel
        if np.random.random() < 0.01:
            temperature_C  += np.random.uniform(15, 35)
        if np.random.random() < 0.01:
            vibration_mm_s += np.random.uniform(0.5, 1.5)

    return {
        "temperature_C":   round(temperature_C, 1),
        "vibration_mm_s":  round(abs(vibration_mm_s), 3),
        "courant_A":       round(courant_A, 2),
        "pression_bar":    round(pression_bar, 2),
        "vitesse_tr_min":  int(vitesse_tr_min),
    }


def generer_machine(usine_id, machine_id_local, type_machine, dates, climat):
    """
    Génère la série temporelle complète pour une machine.
    Inclut des cycles de vie avec pannes planifiées / non planifiées.

    Corrélation 3 : l'âge de la machine réduit le MTBF
    → une machine vieille tombe en panne plus souvent.
    """
    records = []
    age_initial = age_machine_aleatoire()

    # ── Corrélation 3 : fréquence de pannes selon l'âge ──────────────────────
    # Machine jeune (< 10 000h) : MTBF ≈ 1 440h (2 mois)
    # Machine vieille (50 000h) : MTBF ≈ 720h (1 mois)
    age_factor = 1 + (age_initial / 50000) * 0.8  # 1.0 → 1.8
    mtbf_base  = int(1440 / age_factor)            # heures entre pannes

    # Planification des pannes sur la période
    duree_totale_h = len(dates)
    pannes_planifiees = []
    t = random.randint(200, 1000)
    while t < duree_totale_h:
        duree_panne = random.randint(4, 48)
        pannes_planifiees.append((t, duree_panne))
        # Intervalle suivant : réduit si la machine vieillit
        age_en_cours  = age_initial + t
        af_courant    = 1 + (age_en_cours / 50000) * 0.8
        mtbf_courant  = int(1440 / af_courant)
        intervalle    = int(np.random.normal(mtbf_courant, mtbf_courant * 0.2))
        intervalle    = max(300, intervalle)
        t += intervalle + duree_panne

    # Dernière maintenance (en heures avant le début)
    h_depuis_maintenance = random.randint(0, 600)

    en_panne       = False
    nb_pannes      = 0
    nb_maintenances = 0

    for i, date in enumerate(dates):
        age_courant = age_initial + i

        # Vérifier début / fin de panne
        panne_active = any(debut <= i < debut + duree for debut, duree in pannes_planifiees)

        if panne_active and not en_panne:
            en_panne = True
            nb_pannes += 1
        elif not panne_active and en_panne:
            en_panne = False
            h_depuis_maintenance = 0  # maintenance corrective effectuée
            nb_maintenances += 1

        # Heures restantes avant la prochaine panne (RUL)
        prochaine_panne = None
        for debut, duree in pannes_planifiees:
            if debut > i:
                prochaine_panne = debut - i
                break
        rul = prochaine_panne if prochaine_panne is not None else 720
        rul = min(rul, 720)

        score_deg = simuler_degradation(rul, h_depuis_maintenance)
        # Pendant une panne, le score de dégradation pointe vers la prochaine panne
        # (lointaine) → score ≈ 0. On force score=1.0 pour que les capteurs
        # reflètent une machine en état de défaillance maximale.
        if en_panne:
            score_pour_capteurs = 1.0
        else:
            score_pour_capteurs = score_deg
        capteurs  = simuler_capteurs(
            score_pour_capteurs, type_machine, date, climat,
            h_depuis_maintenance,
            mode_critique=(rul < 48 and not en_panne),
            en_panne=en_panne
        )

        # Étiquette de classification
        if en_panne:
            etat = "en_panne"
        elif rul < 48:
            etat = "critique"     # Panne dans moins de 48h
        elif rul < 168:
            etat = "a_risque"     # Panne dans moins d'1 semaine
        else:
            etat = "normal"

        h_depuis_maintenance += 1

        records.append({
            "timestamp":              date.strftime("%Y-%m-%d %H:%M:%S"),
            "usine_id":               usine_id,
            "machine_id":             f"{usine_id}-M{machine_id_local:02d}",
            "type_machine":           type_machine,
            "age_machine_h":          age_courant,
            "h_depuis_maintenance":   h_depuis_maintenance,
            "temperature_C":          capteurs["temperature_C"],
            "vibration_mm_s":         capteurs["vibration_mm_s"],
            "courant_A":              capteurs["courant_A"],
            "pression_bar":           capteurs["pression_bar"],
            "vitesse_tr_min":         capteurs["vitesse_tr_min"],
            "score_degradation":      score_deg,
            "rul_heures":             rul,
            "etat_machine":           etat,
            "en_panne":               int(en_panne),
            "nb_pannes_total":        nb_pannes,
            "nb_maintenances_total":  nb_maintenances,
        })

    return records


# ─── Défauts de qualité (corrélation 7) ───────────────────────────────────────

def _episodes(rng, n_lignes, n_episodes, duree_min, duree_max):
    """Intervalles [début, fin) aléatoires dans une série de n_lignes."""
    for _ in range(n_episodes):
        duree = int(rng.integers(duree_min, duree_max + 1))
        if n_lignes <= duree:
            continue
        debut = int(rng.integers(0, n_lignes - duree))
        yield debut, debut + duree


def injecter_defauts_qualite(df, seed=SEED + 1):
    """
    Dégrade les mesures capteurs comme le ferait un parc IoT réel.
    Utilise son propre générateur aléatoire : la simulation physique en amont
    reste identique, seuls les défauts s'ajoutent. Les labels sont intacts.
    Retourne (df_brut, bilan des défauts injectés).
    """
    rng = np.random.default_rng(seed)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    pays = df["usine_id"].map(lambda u: USINES[u]["pays"])
    bilan = {}

    # 1. Valeurs manquantes ponctuelles (plus fréquentes sur les sites récents)
    taux = pays.map(DEFAUTS["taux_valeur_manquante"]).to_numpy()
    n_manq = 0
    for col in CAPTEURS:
        masque = rng.random(len(df)) < taux
        df.loc[masque, col] = np.nan
        n_manq += int(masque.sum())
    bilan["valeurs_manquantes"] = n_manq

    # 2. Pertes de transmission (tous les capteurs absents) et 3. capteurs figés
    n_pertes = n_figes = 0
    for _, idx in df.groupby("machine_id").indices.items():
        n = len(idx)
        annees = n / 8760
        for debut, fin in _episodes(rng, n, rng.poisson(DEFAUTS["pertes_transmission_par_an"] * annees),
                                    *DEFAUTS["duree_perte_h"]):
            df.loc[idx[debut:fin], CAPTEURS] = np.nan
            n_pertes += 1
        for debut, fin in _episodes(rng, n, rng.poisson(DEFAUTS["capteurs_figes_par_an"] * annees),
                                    *DEFAUTS["duree_capteur_fige_h"]):
            col = CAPTEURS[int(rng.integers(len(CAPTEURS)))]
            df.loc[idx[debut:fin], col] = df.at[idx[debut], col]
            n_figes += 1
    bilan["pertes_transmission"] = n_pertes
    bilan["episodes_capteur_fige"] = n_figes

    # 4. Valeurs aberrantes (codes d'erreur capteur, hors limites physiques)
    n_aber = 0
    for col in CAPTEURS:
        masque = (rng.random(len(df)) < DEFAUTS["taux_valeur_aberrante"]) & df[col].notna().to_numpy()
        df.loc[masque, col] = rng.choice(VALEURS_ABERRANTES[col], size=int(masque.sum()))
        n_aber += int(masque.sum())
    bilan["valeurs_aberrantes"] = n_aber

    # 5. Doublons de transmission (même machine, même horodatage)
    doublons = df.sample(frac=DEFAUTS["taux_doublon"], random_state=seed)
    df = (pd.concat([df, doublons])
          .sort_values(["machine_id", "timestamp"], kind="stable")
          .reset_index(drop=True))
    bilan["doublons"] = len(doublons)

    return df, bilan


# ─── Génération principale ────────────────────────────────────────────────────

def generer_dataset():
    print("=" * 60)
    print("  MECHA — Génération du jeu de données synthétique")
    print("=" * 60)

    all_records = []

    for usine_id, infos in USINES.items():
        # Génération des timestamps (toutes les heures) depuis le début de collecte du site
        debut = DEBUT_COLLECTE_PAYS[infos["pays"]]
        dates = [debut + timedelta(hours=h)
                 for h in range(int((DATE_FIN - debut).total_seconds() // 3600))]
        print(f"  Usine {usine_id} ({infos['ville']}, {infos['pays']}) — "
              f"{infos['nb_machines']} machines — {debut.date()} → {DATE_FIN.date()}")
        for m in range(1, infos["nb_machines"] + 1):
            type_m = random.choice(TYPE_MACHINES)
            records = generer_machine(usine_id, m, type_m, dates, infos["climat"])
            all_records.extend(records)

    df = pd.DataFrame(all_records)
    print(f"\nDataset simulé : {len(df):,} lignes × {len(df.columns)} colonnes")

    # ── Défauts de qualité : fichier brut tel qu'il sortirait du parc IoT ──
    df_brut, bilan = injecter_defauts_qualite(df)
    print("\nDéfauts injectés dans le fichier brut :")
    for k, v in bilan.items():
        print(f"  {k:<24} : {v:,}")

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_brut.to_csv(SORTIE, index=False)
    print(f"\n  ✓ {SORTIE} sauvegardé ({len(df_brut):,} lignes)")

    # ── Vérification des corrélations (sur les données simulées, avant défauts) ──
    print("\n─── Vérification des corrélations ───────────────────────────────")
    print("\nMoyenne température par état_machine :")
    print(df.groupby("etat_machine")["temperature_C"].mean().sort_values(ascending=False).to_string())

    print("\nMoyenne vibration par état_machine :")
    print(df.groupby("etat_machine")["vibration_mm_s"].mean().sort_values(ascending=False).to_string())

    print("\nMoyenne pression par état_machine :")
    print(df.groupby("etat_machine")["pression_bar"].mean().sort_values().to_string())

    print("\nCorrélation Pearson avec score_degradation :")
    cols = ["temperature_C", "vibration_mm_s", "courant_A", "pression_bar", "vitesse_tr_min"]
    for col in cols:
        r = df[col].corr(df["score_degradation"])
        print(f"  {col:<22} : {r:+.4f}")

    print("\nDistribution des états machines :")
    print(df["etat_machine"].value_counts().to_string())
    print(f"\nTaux de panne global : {df['en_panne'].mean()*100:.2f}%")

    return df


if __name__ == "__main__":
    df = generer_dataset()
    print("\n✅ Génération terminée avec succès !")
