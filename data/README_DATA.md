# 📂 Comment obtenir les données (NinaPro DB2)

## Quel dataset ?
On utilise **NinaPro DB2** : c'est LE jeu de données de référence en robotique
biomédicale pour le contrôle de prothèses par signaux musculaires.

- **Contenu :** 40 sujets sains, 49 gestes de la main + repos.
- **Capteurs :** 12 électrodes EMG (signal musculaire) échantillonnées à 2000 Hz,
  **+ un gant CyberGlove** qui mesure l'angle des doigts (le "vrai" mouvement),
  **+ une centrale inertielle (IMU)**.
- **Pourquoi c'est important :** comme on a À LA FOIS le signal musculaire ET le
  mouvement réel des doigts (synchronisés), on peut vérifier que nos données sont
  cohérentes. C'est ce que le barème appelle "synchronisation des modalités"
  (critère 1) — un point fort difficile à obtenir avec un dataset plus simple.

## Où le télécharger
1. Va sur le site officiel : **https://ninapro.hevs.ch/**
2. Section *"Database 2 (DB2)"*. Une inscription gratuite (email) est demandée.
3. Télécharge au moins **les sujets DB2_s1 à DB2_s5** (5 sujets suffisent pour
   le projet ; télécharger les 40 est inutile et très lourd).
4. Dézippe-les ici, dans ce dossier `data/`. Tu dois obtenir :

```
data/
├── DB2_s1/  S1_E1_A1.mat  S1_E2_A1.mat  S1_E3_A1.mat
├── DB2_s2/  ...
├── DB2_s3/
├── DB2_s4/
└── DB2_s5/
```

> On n'utilisera que les fichiers `*_E1_*.mat` (Exercice 1 = mouvements de base
> des doigts), c'est suffisant et plus rapide.

## Que contient un fichier .mat ?
Chaque fichier est une "boîte" contenant notamment :
- `emg` : le signal des 12 muscles, forme (temps, 12).
- `restimulus` : le geste demandé à chaque instant (0 = repos, 1, 2, 3...).
- `rerepetition` : le numéro de la répétition (le sujet refait chaque geste 6 fois).
- `glove` : les angles des doigts mesurés par le gant (la "vérité terrain").

Le script `src/01_exploration.py` ouvre tout ça pour toi et te l'explique.

## Trop lourd pour ta machine ? (plan B)
Si DB2 est trop gros à télécharger/traiter, **NinaPro DB1** est plus léger
(EMG à 100 Hz, 27 sujets). Dans ce cas, mets `FS = 100` dans `src/utils.py`
et adapte les bornes du filtre passe-bande (ex. 1–45 Hz). Le reste du code
fonctionne pareil. On en discute dans le rapport (critère 1).
