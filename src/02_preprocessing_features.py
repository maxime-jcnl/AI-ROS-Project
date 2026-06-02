"""
02_preprocessing_features.py
============================
Ce script, pour chaque sujet :
  1. ouvre les fichiers .mat ;
  2. ne garde que quelques gestes (qu'on associera à des actions de la pince) ;
  3. nettoie le signal + découpe en fenêtres de 200 ms + extrait les features
     (tout ça via src/utils.py) ;
  4. note de quel sujet vient chaque fenêtre (essentiel pour valider
     "sujet par sujet" plus tard - critère 3) ;
  5. sauvegarde le tout dans data/dataset.npz.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

from utils import sliding_windows, GESTE_NINAPRO_VERS_ACTION, NOMS_ACTIONS  

# ---------------------------------------------------------------------------
# (La table GESTE_NINAPRO_VERS_ACTION / NOMS_ACTIONS est définie dans utils.py
#  pour qu'elle soit IDENTIQUE côté ML et côté ROS2.)
# ---------------------------------------------------------------------------

DATA_DIR = "data"
OUT_FILE = "data/dataset.npz"


def charger_sujet(dossier_sujet):
    """Charge tous les .mat E1 d'un sujet et renvoie (X, y) après features."""
    X_list, y_list = [], []
    fichiers = sorted(glob.glob(os.path.join(dossier_sujet, "*_E1_*.mat")))
    if not fichiers:
        return None, None

    for f in fichiers:
        mat = loadmat(f)
        emg = mat["emg"]                  # (temps, 12)
        stim = mat["restimulus"].ravel()  # (temps,)

        # On ne garde que les instants correspondant aux gestes choisis,
        # et on remplace le label NinaPro par notre label d'action (0-3).
        masque = np.isin(stim, list(GESTE_NINAPRO_VERS_ACTION.keys()))
        emg_sel = emg[masque]
        stim_sel = np.array([GESTE_NINAPRO_VERS_ACTION[s] for s in stim[masque]])

        if len(emg_sel) < 500:
            continue

        Xf, yf = sliding_windows(emg_sel, stim_sel)
        X_list.append(Xf)
        y_list.append(yf)

    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)


def sous_echantillonner_repos(X, y, ratio_max=1.0):
    """
    Réduit le nombre de fenêtres 'REPOS' (action 0) pour rééquilibrer.

    Le repos est naturellement bien plus fréquent que les vrais gestes.
    On garde au plus  ratio_max x (taille du plus gros vrai geste)  fenêtres
    de repos, choisies au hasard. Les vrais gestes (1,2,3) sont tous conservés.

    Exemple : si le plus gros geste actif a 800 fenêtres et ratio_max=1.0,
    on garde au plus 800 fenêtres de repos (au lieu de plusieurs milliers).
    """
    rng = np.random.default_rng(0)  # graine fixe -> reproductible
    idx_repos = np.where(y == 0)[0]
    idx_actifs = np.where(y != 0)[0]
    if len(idx_actifs) == 0 or len(idx_repos) == 0:
        return X, y

    # taille du plus gros geste actif
    tailles_actifs = [np.sum(y == a) for a in np.unique(y[idx_actifs])]
    cible_repos = int(max(tailles_actifs) * ratio_max)

    if len(idx_repos) > cible_repos:
        idx_repos = rng.choice(idx_repos, size=cible_repos, replace=False)

    garde = np.sort(np.concatenate([idx_repos, idx_actifs]))
    return X[garde], y[garde]


def main():
    dossiers = sorted(d for d in glob.glob(os.path.join(DATA_DIR, "DB2_s*"))
                      if os.path.isdir(d))   # ignore les .zip et autres fichiers
    if not dossiers:
        print("Aucun dossier DB2_s* trouvé dans data/.")
        print("   -> Télécharger les données (voir data/README_DATA.md)")
        return

    X_all, y_all, subj_all = [], [], []
    for sid, dossier in enumerate(dossiers, start=1):
        print(f"Traitement de {dossier} ...", end=" ")
        X, y = charger_sujet(dossier)
        if X is None:
            print("aucune donnée exploitable, ignoré.")
            continue
        # --- RÉÉQUILIBRAGE : on réduit la classe REPOS (sur-représentée) ---
        # Le repos (action 0) apparaît entre chaque geste -> il écrase les
        # autres. On le sous-échantillonne (undersampling) pour qu'il ne
        # dépasse pas 1.5x le plus gros des VRAIS gestes : on laisse le repos
        # un peu majoritaire (c'est réaliste : le repos doit aussi être
        # reconnu) sans pour autant qu'il écrase les autres classes.
        # On le fait PAR SUJET pour ne pas mélanger les sujets (important
        # pour la validation au §3).
        X, y = sous_echantillonner_repos(X, y, ratio_max=1.5)
        X_all.append(X)
        y_all.append(y)
        subj_all.append(np.full(len(y), sid))  # on retient l'identifiant du sujet
        print(f"{len(y)} fenêtres conservées (après rééquilibrage).")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    subjects = np.concatenate(subj_all)

    print("\n=== RÉSUMÉ DU DATASET CONSTRUIT ===")
    print(f"  Total fenêtres : {len(y)}")
    print(f"  Taille d'une fenêtre (nb de features) : {X.shape[1]}")
    print("  Répartition par action :")
    for a in sorted(np.unique(y)):
        n = np.sum(y == a)
        print(f"    {NOMS_ACTIONS[int(a)]:7s} : {n:6d} ({100*n/len(y):4.1f} %)")

    np.savez_compressed(OUT_FILE, X=X, y=y, subjects=subjects)
    print(f"\nDataset sauvegardé dans {OUT_FILE}")
    


if __name__ == "__main__":
    main()
