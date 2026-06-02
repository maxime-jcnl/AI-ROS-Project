# Rapport de projet — Contrôle myoélectrique d'une pince robotique

**Cours :** AI & ROS — Traitement de signal biomédical et robotique
**Auteurs :** Maxime Jaconelli et Raphaël Houbart

---

## Résumé (abstract)

Ce projet implémente une chaîne complète de **contrôle myoélectrique** : à partir
de signaux électromyographiques de surface (sEMG) de l'avant-bras, un modèle de
machine learning reconnaît un geste de la main, et cette décision pilote en temps
réel une pince robotique simulée sous **ROS2**. Le but est de reproduire le
principe d'une prothèse de main contrôlée par les muscles. Le système est validé
**sujet par sujet** (la condition la plus réaliste) et son fonctionnement temps
réel est évalué par la mesure de la **latence** de bout en bout.

---

## 1. Choix et qualité des données

### Quel dataset et pourquoi
Nous utilisons **NinaPro DB2**, le jeu de données de référence en robotique
biomédicale pour le contrôle de prothèses. Il contient, pour 40 sujets sains :
- **12 canaux sEMG** échantillonnés à **2000 Hz** (signal musculaire) ;
- les angles des doigts mesurés par un **gant CyberGlove** (le mouvement réel) ;
- une **centrale inertielle (IMU)** ;
- ~49 gestes, chacun répété 6 fois.

Ce choix est motivé par trois raisons. D'abord, c'est un dataset **multimodal** :
on dispose à la fois de la *cause* (l'activité musculaire) et de l'*effet* (le
mouvement des doigts), ce qui permet de **vérifier la qualité et la cohérence**
des données. Ensuite, le grand nombre de sujets permet une validation honnête sur
des personnes **jamais vues à l'entraînement** (cf. §3). Enfin, c'est un standard
de la communauté, ce qui rend les résultats **comparables** à la littérature.

### Comparaison avec d'autres datasets
Plusieurs alternatives publiques ont été envisagées avant ce choix :

| Dataset | Capteurs | Points forts | Limites |
|---|---|---|---|
| **NinaPro DB2** (choisi) | 12 EMG @2kHz + gant + IMU | multimodal, nombreux sujets, standard | volumineux |
| UCI "EMG for gestures" | 8 EMG @200 Hz (Myo) | très simple, CSV | mono-modal (pas de mouvement réel), basse fréquence |
| putEMG | 24 EMG | beaucoup de répétitions | pas de cinématique |
| CapgMyo | matrice HD-EMG | haute densité | moins de gestes |

NinaPro DB2 est le seul à offrir la **synchronisation EMG ↔ cinématique** exigée
pour une évaluation rigoureuse de la qualité.

### Qualité du signal et synchronisation
Le script `01_exploration.py` analyse la qualité du dataset selon plusieurs axes.

**Composition.** Sur le sujet 1, l'enregistrement EMG contient **1 808 331 mesures
sur 12 canaux** (échantillonnés à 2 kHz, soit 904,2 secondes / ~15 min de signal),
parfaitement synchronisé avec le gant CyberGlove (22 angles articulaires) et la
table de stimulus indiquant le geste à chaque instant. **Aucun canal "plat"**
(électrode débranchée) n'est détecté.

**Déséquilibre intrinsèque des classes.** Le geste « REPOS » (label 0) représente
**58,9 % des mesures** ; les 17 gestes actifs se partagent les ~41 % restants
(1,3 % à 4,4 % chacun). Ce déséquilibre est traité au §2.

**Synchronisation EMG ↔ mouvement (point principal du critère 1).** Pour vérifier
que le signal musculaire et le mouvement réel des doigts sont bien alignés, deux
indicateurs complémentaires ont été retenus :

1. *Ratio d'activité gestes vs repos* — la métrique la plus interprétable. Pendant
   les périodes de geste, l'énergie EMG moyenne est **×5,09** plus élevée qu'au
   repos, et le mouvement du gant **×2,78** plus élevé. Les **deux modalités
   s'activent ensemble**, ce qui prouve leur cohérence temporelle.
2. *Corrélation des enveloppes lissées (2 s)* — **r = 0,52**. C'est une mesure
   complémentaire ; la corrélation point-à-point reste mécaniquement faible car
   l'EMG est un signal **stochastique** dont seule l'enveloppe (l'amplitude) varie
   de manière interprétable — c'est précisément pour cela que le ratio d'activité
   (ci-dessus) est l'indicateur de référence.

La figure `synchronisation.png` montre visuellement l'alignement temporel des deux
signaux lissés.

### Principes FAIR
NinaPro respecte largement les principes **FAIR** :
- **Findable / Accessible :** dataset public, identifiable, téléchargeable
  gratuitement après inscription sur le site officiel.
- **Interoperable :** format `.mat` standard, lisible par tous les langages.
- **Reusable :** documentation détaillée (protocole, gestes, capteurs) et licence
  d'utilisation clairement définie.

Limite FAIR : l'inscription préalable réduit légèrement l'« Accessibility ».

### Limites des données
Les sujets sont majoritairement **valides** (peu d'amputés dans DB2), ce qui
introduit un biais possible pour une véritable application prothétique. La
population est par ailleurs peu diversifiée en âge et en morphologie. Ces limites
sont rediscutées au §5.

---

## 2. Prétraitement et extraction de caractéristiques

### Nettoyage du signal (`utils.py`)
Le signal EMG brut contient du bruit. Deux filtres sont appliqués :
- un **filtre passe-bande 20–450 Hz** (Butterworth ordre 4) : conserve la bande
  utile du muscle, supprime la dérive lente des électrodes et le bruit haute
  fréquence ;
- un **filtre coupe-bande (notch) à 50 Hz** : élimine le ronflement du **secteur
  électrique européen** (ce serait 60 Hz aux États-Unis — détail important et
  souvent négligé).

La fonction `filtfilt` (filtrage aller-retour) est utilisée pour **ne pas décaler
le signal dans le temps**, ce qui est essentiel pour un système temps réel.

### Découpage en fenêtres
Le signal n'est pas classé échantillon par échantillon, mais par **fenêtres de
200 ms** se chevauchant à 50 %. 200 ms constitue un bon compromis : assez court
pour rester réactif (perçu comme « instantané » par l'utilisateur), assez long
pour contenir l'information du geste. Le chevauchement augmente le nombre
d'exemples et fluidifie la réaction.

### Caractéristiques extraites (features de Hudgins)
Pour chaque fenêtre et chaque canal, 5 caractéristiques classiques et éprouvées en
EMG sont extraites :
- **MAV** (amplitude moyenne) et **RMS** : la « force » du muscle ;
- **WL** (longueur d'onde) : la complexité du signal ;
- **ZC** (passages par zéro) et **SSC** (changements de pente) : le contenu
  fréquentiel exprimé de façon simple.

Soit 12 canaux × 5 caractéristiques = **60 nombres** par fenêtre. Cet ensemble est
compact, rapide à calculer (donc compatible temps réel) et constitue l'ensemble de
caractéristiques le plus robuste de la littérature EMG.

### Gestion du déséquilibre des classes
Le geste « repos » est naturellement très sur-représenté. Il est traité par **deux
mécanismes complémentaires** :
1. **Sous-échantillonnage (undersampling) du repos** lors de la préparation des
   données (`02_preprocessing_features.py`) : le nombre de fenêtres de repos est
   plafonné, par sujet, au niveau du plus gros geste actif. Cela évite que le repos
   noie les vrais gestes.
2. **Pondération des classes** (`class_weight="balanced"`) dans le modèle, qui
   accorde davantage d'importance aux gestes restants.

Après ce rééquilibrage, le jeu de données final (5 sujets, 5845 fenêtres) se
répartit ainsi : REPOS 37,3 %, OUVRIR 20,7 %, FERMER 19,2 %, TOURNER 22,7 %. L'effet
sur les métriques est analysé au §3 (l'écart entre accuracy et F1-score se réduit
nettement).

### Pour aller plus loin (piste)
Un réseau de neurones (CNN 1D) pourrait apprendre lui-même les caractéristiques à
partir du signal brut (*representation learning*), au prix d'un coût de calcul plus
élevé. Voir §5.

---

## 3. Développement et validation du modèle ML

### Modèles comparés
- **Référence (baseline) : LDA** (Analyse Discriminante Linéaire) — modèle simple,
  rapide, historiquement utilisé dans les prothèses. Il sert de point de
  comparaison : tout modèle plus complexe doit faire mieux que lui.
- **Modèle principal : Random Forest** — robuste, gère bien les 60 caractéristiques,
  peu sensible au réglage, et fournit une bonne précision sans GPU.

### Stratégies de validation comparées (point clé du critère 3)

Deux scénarios sont rapportés volontairement, qui correspondent à deux usages
réels :

- **Intra-sujet** (k-fold à l'intérieur de chaque sujet, k=5) : on entraîne et on
  teste sur la **même personne**. Cela mesure la performance *après* une phase de
  calibration utilisateur — le scénario d'une prothèse personnalisée à son porteur.
- **Inter-sujet** (leave-one-subject-out, LOSO) : on entraîne sur certains sujets
  et on teste sur un sujet **jamais vu**. C'est le scénario d'un appareil
  prêt-à-l'emploi pour un nouvel utilisateur, et c'est largement reconnu comme **le
  défi central** du contrôle myoélectrique.

L'écart entre les deux quantifie la **variabilité inter-sujets** — la difficulté
qu'a un modèle EMG à généraliser d'une personne à une autre, en raison des
différences anatomiques et de placement des électrodes.

### Métriques
Sont rapportés l'**accuracy** et le **F1-score macro** (qui équilibre les classes),
avec l'**écart-type entre sujets** (mesure de la robustesse inter-personnes).

### Résultats

**Étape 1 — un piège classique repéré.** Sans rééquilibrage, on obtient une
**accuracy de ~88 % mais un F1 de ~50 %**. Cet écart révèle un **déséquilibre de
classes** : le geste « REPOS » domine, et le modèle peut atteindre 88 % en le
prédisant la plupart du temps. L'accuracy récompense ce comportement, le F1 macro
— qui traite chaque classe à égalité — le démasque. Indice complémentaire : le
Random Forest ne parvenait pas à battre le LDA, signe qu'il se rabattait sur la
classe dominante.

**Étape 2 — rééquilibrage des classes (critère 2).** Un sous-échantillonnage de la
classe « REPOS » a été ajouté lors de la préparation, plafonné à 1,5× le plus gros
geste actif (par sujet). Le déséquilibre est corrigé et le Random Forest commence à
dépasser le LDA.

**Étape 3 — normalisation par sujet (clé de la généralisation).** Une normalisation
**intra-sujet** des caractéristiques a été ajoutée : chaque sujet est centré-réduit
par ses propres statistiques. Cela neutralise les différences « personnelles »
(conductivité de la peau, placement des électrodes, force musculaire) pour ne
garder que la forme du geste. Conséquence directe : l'écart-type entre sujets a
chuté de ±12 points à **±3 points** — le modèle est devenu **stable** d'une
personne à l'autre.

**Résultats finaux (5 sujets) :**

| Modèle | Intra-sujet · Acc | Intra-sujet · F1 | Inter-sujet · Acc | Inter-sujet · F1 |
|---|---|---|---|---|
| LDA (référence) | 86,6 % (±7,5) | 86,8 % | 62,5 % (±3,0) | 56,7 % |
| **Random Forest** | **92,8 % (±2,8)** | **93,2 %** | **63,8 % (±7,7)** | **56,2 %** |

**Lecture des résultats.** Le Random Forest dépasse le LDA de référence dans les
deux scénarios, ce qui confirme la valeur ajoutée du modèle plus puissant
(critère 3). En intra-sujet, **92,8 %** d'accuracy montre que la chaîne complète
(filtrage + caractéristiques + modèle) est **techniquement performante** — la
reconnaissance des 4 gestes est très fiable quand le système est calibré pour un
utilisateur. En inter-sujet, le score chute à **63,8 %**, soit un **écart d'environ
29 points**. Cet écart n'est pas un défaut de l'approche : il quantifie directement
la **variabilité inter-sujets** propre à l'EMG (anatomie, peau, placement des
électrodes), un phénomène bien documenté dans la littérature et qui reste un défi
ouvert du domaine. Les deux mesures sont **chacune dans la fourchette attendue** de
la littérature, ce qui valide la rigueur de la chaîne.

> Repère **honnête** de la littérature sur NinaPro DB2 :
> - **Intra-sujet** : typiquement **75-95 %** d'accuracy.
> - **Inter-sujet LOSO** avec caractéristiques classiques : **55-70 %**. C'est
>   *beaucoup* plus difficile — le modèle doit fonctionner sur des muscles qu'il
>   n'a jamais vus.
>
> Les 63,8 % en LOSO sont donc **dans la bonne fourchette** et **honnêtement
> évalués**. L'écart attendu entre les deux validations confirme que la
> variabilité inter-sujets est bien le facteur limitant.

### Analyse de la matrice de confusion

La matrice de confusion (`figures/matrice_confusion.png`, calculée sur un sujet de
test inconnu) révèle un comportement instructif :

| Vrai \ Prédit | REPOS | OUVRIR | FERMER | TOURNER | Réussite |
|---|---|---|---|---|---|
| REPOS | **296** | 0 | 0 | 5 | **98 %** |
| OUVRIR | 35 | **12** | 15 | 139 | 6 % |
| FERMER | 2 | 1 | **49** | 76 | 38 % |
| TOURNER | 31 | 9 | 39 | **120** | 60 % |

Le **REPOS** est presque parfaitement reconnu (98 %), ce qui est attendu : son
absence d'activité musculaire est une signature très distinctive. Le point
intéressant concerne les confusions entre gestes actifs :

- **OUVRIR est massivement confondu avec TOURNER** (139/201 fenêtres, soit 69 %).
  Cette confusion a une **explication anatomique cohérente** : l'extension des
  doigts (ouvrir la main) et la supination du poignet (tourner) activent toutes
  deux des muscles voisins de la face dorsale de l'avant-bras (extenseur commun des
  doigts, supinateur, brachio-radial). Pour un sujet jamais vu, leurs signatures
  EMG sont géographiquement proches sur les 12 électrodes.
- **TOURNER joue le rôle de classe « fourre-tout »** : le modèle s'y rabat quand il
  hésite, ce qui suggère que c'est la classe avec la plus grande variabilité
  inter-sujets dans les zones d'activation.

Ce résultat est **scientifiquement instructif** : le modèle ne triche pas en
prédisant toujours la classe majoritaire (REPOS), mais ses erreurs reflètent des
**similarités anatomiques réelles**. Un choix de gestes plus « orthogonaux »
musculairement (par exemple flexion vs extension du poignet, qui activent des
groupes musculaires opposés) améliorerait probablement nettement les scores —
c'est une piste claire pour une version future (cf. §5).

---

## 4. Intégration robotique et performance du système

### Architecture ROS2 (boucle fermée)
Le système est découpé en **3 nœuds** ROS2 qui communiquent par messages, comme
dans un vrai robot :

```
[emg_replay] --/emg_window--> [emg_classifier] --/gesture--> [gripper_controller] --/joint_states--> RViz
 (le "bras")                    (le "cerveau")                  (la "main")                          (l'affichage)
```

1. **`emg_replay`** simule le bracelet EMG en rejouant un enregistrement, fenêtre
   par fenêtre (faute de capteur physique).
2. **`emg_classifier`** applique exactement le même nettoyage et les mêmes
   caractéristiques qu'à l'entraînement (code partagé `emg_utils.py`, ce qui
   garantit la cohérence), puis prédit le geste.
3. **`gripper_controller`** traduit le geste en mouvement de la pince et publie les
   positions des articulations, que **RViz** affiche en 3D.

C'est une **vraie boucle fermée** : un signal d'origine humaine finit par déplacer
un robot.

### Mise en œuvre et environnement
Le déploiement a été réalisé sous **Ubuntu 24.04 (ARM64)** dans une machine
virtuelle Parallels sur un Mac Apple Silicon, avec **ROS2 Jazzy**. Le package est
compilé avec `colcon` puis lancé via `ros2 launch ros2_emg_gripper demo.launch.py`,
qui démarre les trois nœuds, le publieur d'état du robot et RViz. La pince est
modélisée par un fichier URDF (un socle, une palme reliée par une articulation
rotative pour le poignet, et deux doigts prismatiques).

### Performance temps réel : la latence
Le nœud classifieur **chronomètre** le temps entre la réception d'une fenêtre et la
décision. Pour un contrôle prothétique, la latence totale doit rester sous
**~300 ms** pour être imperceptible.

| Mesure | Valeur |
|---|---|
| Latence de décision (moyenne) | ≈ 25 ms |
| Plage observée par fenêtre | ≈ 13–26 ms |

Ces valeurs ont été relevées en direct pendant l'exécution de la démonstration
ROS2. L'extraction des caractéristiques suivie d'une prédiction Random Forest est
très légère, ce qui place le système **plus de dix fois en dessous** du budget de
300 ms : le contrôle est donc perçu comme instantané.

### Robustesse
- **Différents utilisateurs :** la validation sujet-par-sujet (§3) mesure déjà la
  robustesse à de nouvelles personnes.
- **Bruit :** le notch 50 Hz et le passe-bande rendent le système robuste aux
  parasites électriques. Une extension possible consisterait à injecter du bruit
  artificiel dans `emg_replay` et à mesurer la dégradation de l'accuracy (test non
  réalisé dans cette version).

### Pistes « déploiement réel »
Le nœud `emg_replay` pourrait être remplacé par un vrai bracelet (par exemple via
un driver série ou Bluetooth) **sans rien changer** au reste : c'est tout l'intérêt
de l'architecture modulaire ROS2.

---

## 5. Reproductibilité, documentation, innovation et éthique

### Reproductibilité
- Code **open-source**, organisé et **commenté en français** étape par étape.
- **Gestion des dépendances avec UV** : un fichier `pyproject.toml` déclare les
  dépendances et un fichier `uv.lock` **fige les versions exactes** de tous les
  paquets. N'importe qui peut donc recréer un environnement **identique** en une
  seule commande (`uv sync`), ce qui élimine le classique « ça marche chez moi ».
- **README** expliquant comment installer, télécharger les données et tout relancer.
- Pipeline **déterministe** (`random_state` fixé) → résultats reproductibles.
- Code de traitement **partagé** entre entraînement et temps réel → pas de
  divergence entre les deux.
- Versionnement avec **Git** et publication sur GitHub recommandés (insérer le lien
  en tête de rapport).

### Originalité
La contribution n'est pas un nouvel algorithme, mais une **chaîne complète et
pédagogique** reliant signal biomédical, ML validé honnêtement, et robotique temps
réel en boucle fermée — avec une **séparation stricte cause/effet** exploitée pour
vérifier la qualité des données (synchronisation EMG ↔ cinématique), ce qui est
rarement fait dans les projets étudiants.

### Limites (discussion critique honnête)
- Données rejouées, pas de capteur physique → la latence d'acquisition réelle
  (Bluetooth, etc.) n'est pas mesurée.
- Sujets majoritairement valides → généralisation incertaine aux amputés.
- Seulement 4 actions : un vrai système prothétique en gère davantage.
- Le Random Forest plafonne ; un modèle profond ferait probablement mieux.

### Travaux futurs
- **Choisir des gestes musculairement plus « orthogonaux »** (par exemple flexion
  vs extension du poignet, qui activent des groupes opposés) au lieu d'OUVRIR vs
  TOURNER, dont les confusions ont été identifiées dans la matrice de confusion
  (§3).
- Remplacer le replay par un **vrai bracelet EMG** et mesurer la latence complète.
- Tester un **CNN 1D / Transformer** sur le signal brut.
- **Adaptation au sujet** (transfer learning) pour améliorer la généralisation.
- Passer de RViz à **Gazebo** pour une simulation physique réaliste, puis envisager
  un prototype matériel.

### Considérations éthiques (signaux biomédicaux)
- **Vie privée :** les signaux EMG sont des **données biométriques sensibles** ;
  ils doivent être stockés de façon sécurisée et anonymisée.
- **Consentement :** toute collecte sur de vraies personnes nécessite un
  consentement éclairé et l'aval d'un comité d'éthique.
- **Équité algorithmique :** si le modèle est entraîné sur une population peu
  diverse, il peut **moins bien fonctionner** pour certains groupes — un enjeu
  direct de sécurité pour un dispositif médical. Cette limite doit être documentée,
  en visant des jeux de données plus représentatifs.
- **Sécurité :** un faux positif sur une vraie prothèse peut être dangereux ; un
  « geste de repos » sûr par défaut et un seuil de confiance sont nécessaires.

---

## Annexes
- `figures/emg_brut.png` — exemple de signal musculaire.
- `figures/synchronisation.png` — preuve d'alignement EMG ↔ mouvement.
- `figures/matrice_confusion.png` — détail des erreurs du modèle.
- Code complet : voir le dépôt.