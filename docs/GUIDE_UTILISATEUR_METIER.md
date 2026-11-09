# Guide Utilisateur Métier — Solution de Maintenance Prédictive MECHA

**Destinataires** : responsables maintenance, responsables production, pilotage industriel

---

## Ce que fait la solution

La solution analyse chaque heure les relevés des capteurs de vos machines (température, vibration, courant, pression, vitesse de rotation) et vous indique :

- **le risque de panne** de chaque machine, sous forme de probabilité et de niveau de risque ;
- **des alertes** quand une machine reste à un niveau de risque critique plusieurs heures de suite ;
- **un ordre de grandeur du temps restant** avant défaillance (RUL) ;
- **les comportements capteurs inhabituels** (détection d'anomalies), même pour un type de panne jamais rencontré.

C'est un outil d'**aide à la décision** : il vous aide à prioriser les inspections, il ne remplace ni l'expertise des techniciens ni les contrôles réglementaires.

---

## Le tableau de bord

Accessible depuis un navigateur à l'adresse fournie par la DSI (par défaut `http://votre-serveur:8501`).

| Page | Ce qu'elle montre | Pour qui |
|---|---|---|
| **Vue d'ensemble** | État du parc par usine, performance de la règle d'alerte, qualité des données | Direction industrielle, pilotage |
| **Prédiction en temps réel** | Saisie des valeurs d'une machine (ou import d'un CSV de plusieurs machines) → risque, recommandation, RUL et explication | Responsables maintenance, techniciens |
| **Analyse des capteurs** | Historique des capteurs d'une machine, pannes passées, corrélations | Ingénieurs fiabilité |
| **Performance des modèles** | Résultats de validation des modèles et de la règle d'alerte | DSI, équipe data |
| **Monitoring Temps Réel** | **Alertes ouvertes**, jauge de risque par machine (filtrable par usine), dernières prédictions ; rafraîchie toutes les 5 secondes | Responsables maintenance, techniciens |

---

## Lire le niveau de risque

| Niveau | Probabilité de panne dans les 24 h | Action recommandée |
|---|---|---|
| **Faible** (vert) | moins de 30 % | Aucune action |
| **Moyen** (jaune) | 30 à 50 % | Surveillance renforcée, vérifier les paramètres |
| **Élevé** (orange) | 50 à 80 % | Planifier une maintenance préventive |
| **Critique** (rouge) | 80 % et plus | Inspection rapide de la machine |

Un relevé isolé au niveau élevé ou critique **n'est pas une alerte** : les capteurs sont bruités et un pic ponctuel est fréquent.

---

## Lire et traiter une alerte

**Quand une alerte est-elle créée ?** Quand **3 relevés horaires consécutifs** d'une machine sont au niveau **critique**. Une alerte « anomalie » est créée de la même façon quand le comportement des capteurs est très inhabituel 3 heures de suite. Tant qu'une alerte est ouverte, aucune nouvelle alerte n'est créée pour la même machine.

**Que signifie une alerte ?** Une **machine à inspecter en priorité**. D'après la validation :
- 9 alertes sur 10 sont suivies d'une panne **dans la semaine**, mais seulement 1 sur 4 dans les 24 h ;
- 3 pannes sur 4 sont précédées d'une alerte dans les 24 h qui les précèdent ;
- comptez en moyenne une alerte toutes les deux semaines par machine.

Traitez donc une alerte comme « **intervention à planifier dans les prochains jours** », pas comme « panne certaine demain ».

**Démarche recommandée :**
1. Ouvrir la page **Monitoring Temps Réel** et repérer la machine, son niveau et le message.
2. Vérifier les **tendances** dans **Analyse des capteurs** : température ou vibration en hausse, pression ou vitesse en baisse.
3. Consulter **Pourquoi cette prédiction ?** (page Prédiction en temps réel) : les 3 mesures qui pèsent le plus. En rouge, elles augmentent le risque ; en vert, elles le diminuent.
4. Croiser avec l'historique : date de la dernière maintenance, âge de la machine, production en cours.
5. Décider de l'intervention, puis cliquer sur **Acquitter** : l'alerte est fermée et la machine peut à nouveau alerter si le risque persiste.

Acquitter une alerte signifie « prise en charge », pas « fausse alerte ». Notez le résultat de l'inspection dans la GMAO : ces retours servent à calibrer les seuils pendant le pilote.

---

## Le temps restant avant panne (RUL)

Le RUL est un **ordre de grandeur** : son erreur moyenne est d'environ **100 heures**. Il sert à **comparer** les machines entre elles (une machine à ~50 h est plus urgente qu'une machine à ~500 h), **pas à planifier une intervention à l'heure près**. Le RUL ne déclenche jamais d'alerte à lui seul.

---

## Relevés rejetés

Un relevé dont une valeur est physiquement impossible (par exemple une température de 999 °C, typique d'un capteur en erreur) est **rejeté** par la solution et n'est pas analysé. Un relevé incomplet n'est pas transmis. Si une machine n'apparaît plus dans le monitoring, signalez-le à l'équipe instrumentation.

---

## Limites et conditions d'usage

- La solution ne voit que les **capteurs** : un capteur défaillant ou mal calibré fausse les prédictions.
- Elle a appris sur des **données historiques** (simulées pour ce prototype) : un mode de panne jamais observé peut passer inaperçu, d'où la détection d'anomalies en complément.
- Elle travaille sur le **dernier relevé** de chaque machine : elle repère bien une machine qui se dégrade, moins bien le moment exact de la panne.
- Les probabilités ne sont **pas des certitudes** : un risque « faible » ne signifie pas un risque nul.
- Les seuils et la règle d'alerte sont réglables par la DSI dans le fichier `config/regles_metier.yaml`, après validation par les responsables maintenance.

---

## En cas de problème

| Situation | Contact |
|---|---|
| Dashboard inaccessible ou page Monitoring vide | DSI — support IT |
| Machine absente du monitoring, relevés rejetés à répétition | Équipe instrumentation / IoT |
| Alerte qui semble incohérente | Ingénieur fiabilité + équipe data |
| Question sur l'interprétation d'une alerte | Votre responsable maintenance / référent du site |
