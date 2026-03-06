# Guide Utilisateur Métier - Solution de Maintenance Prédictive MECHA

**Destinataires** : Responsables maintenance, Responsables production, Pilotage industriel

---

## Qu'est-ce que cette solution ?

La solution de maintenance prédictive MECHA est un outil d'aide à la décision basé sur l'intelligence artificielle. Elle analyse en continu les données de vos capteurs machines (température, vibration, pression, vitesse de rotation) pour :

- **Détecter les risques de panne** avant qu'elles ne surviennent (jusqu'à 24h à l'avance)
- **Estimer le temps restant** avant une défaillance probable (RUL - Remaining Useful Life)
- **Prioriser les interventions** de maintenance en fonction du niveau de risque

L'objectif est simple : **passer d'une maintenance corrective (on répare quand ça casse) à une maintenance prédictive (on intervient juste avant que ça casse)**.

---

## Comment consulter les résultats ?

### Le tableau de bord (Dashboard)

Le dashboard est accessible via un navigateur web à l'adresse fournie par votre DSI (par défaut : `http://votre-serveur:8501`).

Il comporte 4 vues :

| Vue | Ce qu'elle montre | Pour qui |
|-----|-------------------|----------|
| **Vue d'ensemble** | État global du parc machines, nombre d'alertes, taux de panne | Direction industrielle, pilotage |
| **Prédiction temps réel** | Saisir les valeurs capteurs d'une machine et obtenir une prédiction immédiate | Techniciens maintenance |
| **Analyse des données** | Historique capteurs par machine, tendances, corrélations | Ingénieurs fiabilité |
| **Performance modèles** | Métriques de précision des modèles IA | DSI, équipe data |

### Comprendre les niveaux de risque

Chaque machine se voit attribuer un **niveau de risque** basé sur la probabilité de panne calculée par l'IA :

| Indicateur | Probabilité | Signification | Action recommandée |
|------------|-------------|---------------|-------------------|
| **FAIBLE** (vert) | < 30% | Fonctionnement normal | Aucune intervention nécessaire |
| **MOYEN** (jaune) | 30% - 50% | Dégradation possible détectée | Renforcer la surveillance, vérifier les paramètres |
| **ÉLEVÉ** (orange) | 50% - 80% | Panne probable sous 24h | Planifier une intervention de maintenance préventive |
| **CRITIQUE** (rouge) | > 80% | Panne imminente | **Arrêt recommandé**, intervention urgente requise |

### Comprendre le RUL (temps restant)

Le RUL (Remaining Useful Life) indique le **nombre d'heures estimé avant la prochaine défaillance**. Par exemple :
- RUL = 48h → la machine peut continuer à fonctionner, maintenance à planifier sous 2 jours
- RUL = 6h → intervention à prévoir dans les prochaines heures
- RUL < 2h → arrêt immédiat recommandé

**Attention** : le RUL est une estimation statistique, pas une certitude. Une marge de sécurité de ±1h30 est recommandée.

---

## Comment interpréter une alerte ?

Quand la solution génère une alerte (niveau élevé ou critique), voici la démarche à suivre :

1. **Consulter le dashboard** : identifier la machine concernée et ses données capteurs actuelles
2. **Vérifier les tendances** : dans l'onglet "Analyse des données", observer si les capteurs montrent une dérive récente (augmentation de température, vibrations anormales, etc.)
3. **Croiser avec l'historique** : vérifier la date de la dernière maintenance et l'âge de la machine
4. **Décider** : selon le contexte de production, planifier l'intervention ou renforcer la surveillance

---

## Limites et conditions d'usage

- La solution fonctionne **à partir des données capteurs** : si un capteur est défaillant ou envoie des données erronées, la prédiction sera faussée
- Le modèle a été entraîné sur des **données historiques** : un nouveau type de panne jamais observé ne sera pas détecté
- Les prédictions sont des **probabilités**, pas des certitudes : un risque "faible" ne signifie pas zéro risque
- La solution **ne remplace pas** l'expertise des techniciens : elle est un outil d'aide à la décision complémentaire
- Pour être fiable, la solution nécessite un **historique de 24h minimum** de données capteurs par machine

---

## En cas de problème

| Situation | Contact |
|-----------|---------|
| Dashboard inaccessible | DSI - Support IT |
| Données capteurs manquantes | Équipe instrumentation / IoT |
| Alerte semble incohérente | Ingénieur fiabilité + Équipe data |
| Question sur l'interprétation | Votre responsable maintenance |
