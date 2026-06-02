"""
utils.py
========
Boîte à outils PARTAGÉE par tout le projet.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ---------------------------------------------------------------------------
# PARAMÈTRES GLOBAUX 
# ---------------------------------------------------------------------------
FS = 2000             # Fréquence d'échantillonnage du NinaPro DB2 = 2000 Hz
                      # (= 2000 mesures par seconde sur chaque capteur)
MAINS_FREQ = 50       # Fréquence du courant secteur en Europe = 50 Hz.
                      # (Aux USA ce serait 60 Hz.) On l'éliminera car elle
                      # "pollue" le signal musculaire.
WINDOW_MS = 200       # Taille d'une "fenêtre" d'analyse = 200 millisecondes.
                      # On ne classe pas échantillon par échantillon, mais
                      # par petits morceaux de 200 ms (assez court pour rester
                      # "temps réel", assez long pour voir le geste).
OVERLAP = 0.5         # Les fenêtres se chevauchent à 50 % -> plus d'exemples
                      # et une réaction plus fluide.

WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)   # 200 ms * 2000 Hz = 400 mesures
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))  # on avance de 200 mesures

# ---------------------------------------------------------------------------
# TABLE DES GESTES (source unique, utilisée par le ML ET par ROS2)
# ---------------------------------------------------------------------------
# NinaPro contient ~50 gestes ; on n'en garde que 4, renommés 0..3, et on leur
# donne un sens "robotique". Modifie les numéros NinaPro à gauche si tu veux
# (regarde la sortie de 01_exploration.py pour les labels disponibles).
GESTE_NINAPRO_VERS_ACTION = {
    0: 0,   # repos    -> pince immobile
    5: 1,   # un geste -> ouvrir la pince
    6: 2,   # un autre -> fermer la pince
    9: 3,   # encore   -> tourner le poignet
}
NOMS_ACTIONS = {0: "REPOS", 1: "OUVRIR", 2: "FERMER", 3: "TOURNER"}


# ---------------------------------------------------------------------------
# 1) FILTRAGE DU SIGNAL  (= "nettoyage")
# ---------------------------------------------------------------------------
def _design_filters():
    """On prépare deux filtres (calculés une seule fois)."""
    # Filtre passe-bande 20-450 Hz : on garde uniquement les fréquences où se
    # trouve l'information musculaire utile, on jette le reste (dérive lente
    # des électrodes en dessous de 20 Hz, bruit haute fréquence au-dessus).
    nyq = FS / 2.0
    b_bp, a_bp = butter(4, [20 / nyq, 450 / nyq], btype="band")
    # Filtre "coupe-bande" (notch) sur 50 Hz : retire le ronflement du secteur.
    b_n, a_n = iirnotch(MAINS_FREQ, Q=30, fs=FS)
    return (b_bp, a_bp), (b_n, a_n)


_BP, _NOTCH = _design_filters()


def filter_emg(signal_2d):
    """
    Nettoie un signal EMG.
    Entrée  : tableau (nb_mesures, nb_canaux)  -> ex (400, 12)
    Sortie  : même forme, mais filtré.
    """
    sig = np.asarray(signal_2d, dtype=float)
    # filtfilt = filtre "aller-retour" : ne décale pas le signal dans le temps.
    sig = filtfilt(_BP[0], _BP[1], sig, axis=0)       # passe-bande 20-450 Hz
    sig = filtfilt(_NOTCH[0], _NOTCH[1], sig, axis=0)  # coupe-bande 50 Hz
    return sig


# ---------------------------------------------------------------------------
# 2) EXTRACTION DES CARACTÉRISTIQUES  (features de Hudgins, les classiques)
# ---------------------------------------------------------------------------
# Au lieu de donner 400 chiffres bruts par canal au modèle, on résume chaque
# fenêtre par 5 nombres "parlants" par canal. C'est plus simple, plus rapide,
# et c'est ce qui marche le mieux en pratique pour l'EMG.

def _mav(x):  # Mean Absolute Value : amplitude moyenne (force du muscle)
    return np.mean(np.abs(x), axis=0)

def _rms(x):  # Root Mean Square : autre mesure d'énergie du signal
    return np.sqrt(np.mean(x ** 2, axis=0))

def _wl(x):   # Waveform Length : longueur totale de la "courbe" (complexité)
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def _zc(x, thr=1e-5):  # Zero Crossings : nb de fois où le signal change de signe
    s = np.sign(x)
    return np.sum((s[:-1] * s[1:] < 0) & (np.abs(np.diff(x, axis=0)) > thr), axis=0)

def _ssc(x, thr=1e-5):  # Slope Sign Changes : nb de changements de pente
    d = np.diff(x, axis=0)
    s = np.sign(d)
    return np.sum((s[:-1] * s[1:] < 0) & (np.abs(np.diff(d, axis=0)) > thr), axis=0)


FEATURE_NAMES = ["MAV", "RMS", "WL", "ZC", "SSC"]  # ordre des features


def extract_features(window_2d):
    """
    Transforme UNE fenêtre nettoyée en un vecteur de caractéristiques.
    Entrée : (nb_mesures, nb_canaux)  ex (400, 12)
    Sortie : vecteur 1D de longueur nb_canaux * 5
             ex 12 canaux * 5 features = 60 nombres.
    """
    x = np.asarray(window_2d, dtype=float)
    feats = np.concatenate([_mav(x), _rms(x), _wl(x), _zc(x), _ssc(x)])
    return feats


# ---------------------------------------------------------------------------
# 3) DÉCOUPAGE EN FENÊTRES (= "windowing")
# ---------------------------------------------------------------------------
def sliding_windows(emg, labels):
    """
    Découpe un long enregistrement en fenêtres de 200 ms qui se chevauchent.
    Pour chaque fenêtre, on garde le label majoritaire (le geste dominant).

    Entrée :
        emg    : (N_total, nb_canaux)  signal brut complet
        labels : (N_total,)            le geste à chaque instant
    Sortie :
        X : liste de vecteurs de features (un par fenêtre)
        y : liste de labels (un par fenêtre)
    """
    X, y = [], []
    for start in range(0, len(emg) - WINDOW_SAMPLES, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES
        win = emg[start:end]
        lab_window = labels[start:end]
        # label majoritaire de la fenêtre
        lab = np.bincount(lab_window.astype(int)).argmax()
        win = filter_emg(win)             # nettoyage
        X.append(extract_features(win))   # résumé en features
        y.append(lab)
    return np.array(X), np.array(y)
