# Contrôle myoélectrique d'une pince robotique (EMG → ML → ROS2)

Ce projet met en œuvre une chaîne complète de contrôle myoélectrique : des
signaux électromyographiques (EMG) de l'avant-bras sont enregistrés, un modèle
de classification reconnaît le geste de la main associé, et une pince robotique
simulée sous ROS2 exécute l'action correspondante (ouverture, fermeture,
rotation). Le principe est celui d'une prothèse de main commandée par l'activité
musculaire.

<a href="https://www.youtube.com/watch?v=ZsZidsTis38" target="_blank">
  <img src="https://img.youtube.com/vi/ZsZidsTis38/hqdefault.jpg" alt="Voir la vidéo" width="400" />
</a>
Le projet est réalisé dans le cadre de l'évaluation **AI & ROS** et couvre deux
volets : le traitement de signal biomédical et l'apprentissage automatique d'une
part, l'intégration robotique sous ROS2 d'autre part.

## Principe

Une contraction musculaire produit un courant électrique de faible amplitude,
mesurable en surface par des électrodes : c'est le signal EMG. Chaque geste de la
main génère une signature électrique distincte. Le système apprend à reconnaître
ces signatures à partir de données étiquetées, puis traduit le geste reconnu en
commande pour le robot.

La chaîne de traitement est la suivante :

```
EMG (12 capteurs) → filtrage → extraction de caractéristiques → classification → ROS2 → commande de la pince
```

## Structure du dépôt

```
projet_emg_ros2/
├── README.md
├── RAPPORT.md                          Rapport répondant aux cinq critères d'évaluation
├── pyproject.toml                      Dépendances Python (gérées par UV)
├── uv.lock                             Versions figées pour la reproductibilité
├── data/
│   └── README_DATA.md                  Procédure de téléchargement des données NinaPro
├── src/                                Traitement du signal et apprentissage (Python)
│   ├── utils.py                        Fonctions partagées : filtres et caractéristiques
│   ├── 01_exploration.py               Exploration et contrôle qualité des données
│   ├── 02_preprocessing_features.py    Préparation du jeu de données
│   └── 03_train_model.py               Entraînement et validation du modèle
└── ros2_emg_gripper/                   Package ROS2 (intégration robotique)
    ├── ros2_emg_gripper/
    │   ├── emg_replay_node.py          Rejeu du signal EMG en temps réel
    │   ├── emg_classifier_node.py      Classification du geste et mesure de latence
    │   └── gripper_controller_node.py  Commande de la pince
    ├── urdf/simple_gripper.urdf        Description géométrique de la pince
    └── launch/demo.launch.py           Lancement de l'ensemble des nœuds
```

## Exécution

### Partie A — Traitement et apprentissage (Python)

Le projet utilise [UV](https://docs.astral.sh/uv/) pour la gestion de
l'environnement et des dépendances. L'installation manuelle de Python ou la
création d'un environnement virtuel ne sont pas nécessaires.

```bash
# Installation de UV (une seule fois)
#   Linux / macOS :
curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows (PowerShell) :
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Création de l'environnement et installation des dépendances
uv sync

# Téléchargement des données (voir data/README_DATA.md), puis exécution des étapes
uv run python src/01_exploration.py             # Exploration et figures
uv run python src/02_preprocessing_features.py  # Préparation du jeu de données
uv run python src/03_train_model.py             # Entraînement et validation
```

La commande `uv run` exécute le script dans l'environnement du projet sans
activation préalable. Le fichier `uv.lock`, généré par `uv sync`, fige les
versions exactes des paquets, ce qui garantit un environnement identique sur
toute machine (critère 5).

Installation alternative avec pip :

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install numpy scipy scikit-learn joblib matplotlib
python src/01_exploration.py
```

Ces étapes produisent le modèle entraîné (`model/emg_model.joblib`), les scores
de validation et les figures (signal brut, synchronisation, matrice de
confusion).

### Partie B — Intégration robotique (ROS2)

ROS2 s'exécute sous Ubuntu. La distribution dépend de la version d'Ubuntu :
**ROS2 Jazzy** pour Ubuntu 24.04, **ROS2 Humble** pour Ubuntu 22.04. Sous Windows
ou macOS, une machine virtuelle Ubuntu (Parallels, UTM, VirtualBox) ou un
conteneur Docker permet d'obtenir l'environnement requis.

```bash
# Création de l'espace de travail
mkdir -p ~/ros2_ws/src
cp -r ros2_emg_gripper ~/ros2_ws/src/

# Compilation
cd ~/ros2_ws
colcon build --packages-select ros2_emg_gripper
source install/setup.bash

# Lancement (depuis le dossier du projet, pour que les chemins data/ et model/ soient résolus)
cd /chemin/vers/projet_emg_ros2
ros2 launch ros2_emg_gripper demo.launch.py
```

Le lancement démarre les trois nœuds, le publieur d'état du robot et RViz. Dans
RViz, ajouter l'affichage *RobotModel* et régler *Fixed Frame* sur `base_link`
pour visualiser la pince. Les terminaux affichent le geste reconnu et la latence
de classification en millisecondes.

Exécution sous Docker :

```bash
docker run -it --rm osrf/ros:humble-desktop bash
# puis suivre les étapes ci-dessus dans le conteneur
```

## Correspondance avec le barème

| Critère | Localisation |
|---|---|
| 1. Choix et qualité des données | `RAPPORT.md` §1 et `01_exploration.py` (synchronisation, contrôle qualité) |
| 2. Prétraitement et caractéristiques | `utils.py` (filtres 20–450 Hz, notch 50 Hz, 5 caractéristiques par canal) et gestion du déséquilibre des classes |
| 3. Modèle et validation | `03_train_model.py` (LDA et Random Forest, validation intra-sujet et inter-sujet) |
| 4. Intégration robotique ROS2 | `ros2_emg_gripper/` (trois nœuds, boucle fermée, mesure de latence) |
| 5. Reproductibilité et discussion | dépôt complet (code documenté, README) et `RAPPORT.md` §5 (limites, éthique, perspectives) |

Le détail méthodologique et les résultats figurent dans `RAPPORT.md`.
