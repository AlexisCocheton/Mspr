# Conduite du Changement - Déploiement de la Solution de Maintenance Prédictive

## 1. Contexte du changement

Le passage d'une maintenance **corrective/préventive planifiée** à une maintenance **prédictive basée sur l'IA** représente un changement significatif dans les pratiques de travail des équipes de production et de maintenance de MECHA.

Ce changement impacte :
- Les **méthodes de travail** : les techniciens doivent intégrer un nouvel outil dans leur routine quotidienne
- Les **processus de décision** : les priorités d'intervention sont désormais guidées par des indicateurs IA
- La **culture d'entreprise** : passage d'une logique réactive à une logique anticipatrice

---

## 2. Acteurs concernés

| Acteur | Rôle dans le changement | Impact |
|--------|------------------------|--------|
| **Direction générale** | Sponsor du projet, valide la stratégie | Faible (décisionnel) |
| **Direction industrielle** | Pilote le déploiement multi-sites | Moyen |
| **Responsables maintenance** (5 sites) | Utilisateurs principaux, adaptent la planification | **Fort** |
| **Techniciens maintenance** (~50 personnes) | Utilisateurs quotidiens du dashboard et des alertes | **Fort** |
| **Responsables production** | Intègrent les alertes dans la gestion des lignes | Moyen |
| **Opérateurs de production** | Informés des arrêts planifiés liés aux prédictions | Faible |
| **DSI** | Maintient l'infrastructure, support technique | Moyen |
| **Direction qualité** | Suit l'impact sur les taux de rebuts et non-conformités | Faible |

---

## 3. Principaux messages

### Pour la Direction
> "La maintenance prédictive permet de réduire les arrêts non planifiés de 30 à 50%, d'optimiser les coûts de maintenance et de renforcer notre compétitivité face à la concurrence internationale."

### Pour les Responsables maintenance
> "Cet outil vous donne une visibilité anticipée sur l'état de vos machines. Il ne remplace pas votre expertise, il la renforce en vous fournissant des indicateurs objectifs pour prioriser vos interventions."

### Pour les Techniciens
> "Le dashboard vous alerte quand une machine risque de tomber en panne. Vous gardez la main sur la décision d'intervenir. L'outil vous aide à mieux planifier votre travail et à réduire les urgences."

---

## 4. Plan de formation

### Phase 1 : Formation des référents (Semaine 1-2)

| Session | Public | Durée | Contenu |
|---------|--------|-------|---------|
| Formation technique | 1 référent maintenance par site + DSI | 1 journée | Architecture, API, administration, dépannage |
| Formation fonctionnelle | Responsables maintenance (5 sites) | ½ journée | Dashboard, interprétation des alertes, processus de décision |

### Phase 2 : Formation des équipes (Semaine 3-4)

| Session | Public | Durée | Contenu |
|---------|--------|-------|---------|
| Prise en main | Techniciens maintenance (par groupes de 10) | 2 heures | Utilisation du dashboard, lecture des niveaux de risque, procédure d'alerte |
| Sensibilisation | Responsables production | 1 heure | Comprendre les alertes, impact sur la planification production |

### Phase 3 : Accompagnement terrain (Semaine 5-8)

- Présence d'un référent formé sur chaque site pendant 2 semaines
- Hotline support (DSI) disponible pour les questions techniques
- Sessions de retour d'expérience hebdomadaires (30 min) pendant 1 mois

### Supports de formation fournis

- Guide utilisateur métier (document livré avec la solution)
- Vidéos tutorielles courtes (2-3 min) pour chaque fonctionnalité du dashboard
- FAQ mise à jour en continu sur l'intranet

---

## 5. Stratégie de déploiement progressif

### Étape 1 : Pilote (Mois 1-2) — 1 site France

- Déploiement sur **un seul site** (le mieux équipé en capteurs IoT)
- Périmètre limité : **10-20 machines** représentatives
- Mode **observation** : les alertes sont générées mais n'impactent pas encore la planification officielle
- Objectif : valider les prédictions, calibrer les seuils, recueillir les retours terrain

### Étape 2 : Extension France (Mois 3-4) — 3 sites France

- Déploiement sur les **3 sites français**
- Mode **opérationnel** : les alertes sont intégrées dans le processus de planification maintenance
- Ajustements des seuils si nécessaire en fonction des retours du pilote
- Mesure des premiers indicateurs de performance (réduction des arrêts non planifiés)

### Étape 3 : Déploiement international (Mois 5-6) — 5 sites

- Extension aux **2 sites espagnols**
- Adaptation linguistique du dashboard si nécessaire (interface multilingue)
- Homogénéisation des pratiques entre tous les sites

### Critères de passage d'une étape à la suivante

- Taux d'adoption du dashboard > 80% des techniciens du site
- Moins de 5 incidents techniques non résolus par semaine
- Retour positif des responsables maintenance sur la pertinence des alertes
- Aucun arrêt de production causé par un faux négatif (panne non détectée)

---

## 6. Gestion des résistances

| Résistance anticipée | Réponse |
|---------------------|---------|
| "L'IA va remplacer mon expertise" | L'outil est un assistant, pas un remplaçant. Le technicien garde la décision finale. |
| "Les prédictions ne sont pas fiables" | Phase pilote en mode observation pour démontrer la fiabilité avant passage en opérationnel. |
| "C'est un outil de plus à gérer" | Le dashboard est conçu pour être simple (4 vues, code couleur intuitif). Formation courte de 2h. |
| "Ça ne marchera pas sur nos machines spécifiques" | Les modèles sont entraînés sur des données proches du contexte MECHA et ajustables par site. |

---

## 7. Indicateurs de succès du changement

| Indicateur | Cible à 6 mois |
|------------|----------------|
| Taux d'adoption du dashboard | > 85% des techniciens consultent le dashboard au moins 1 fois/jour |
| Réduction des arrêts non planifiés | -20% par rapport à la période précédente |
| Temps moyen de réaction à une alerte | < 4 heures |
| Satisfaction utilisateurs (enquête) | Note > 3.5/5 |
| Nombre de faux positifs critiques | < 5% des alertes "critique" |
