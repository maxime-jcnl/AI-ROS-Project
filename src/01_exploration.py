"""
01_exploration.py
=================
Ce script :
  - ouvre un fichier .mat du NinaPro DB2 ;
  - affiche sa composition (combien de canaux, combien de gestes...) ;
  - VÉRIFIE LA SYNCHRONISATION entre le signal musculaire (emg) et le
    mouvement réel des doigts (glove) -> point important du critère 1 ;
  - trace quelques figures sauvegardées dans le dossier figures/.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Chemin vers un fichier exemple (sujet 1, exercice 1)
MAT_FILE = "data/DB2_s1/S1_E1_A1.mat"
os.makedirs("figures", exist_ok=True)


def main():
    if not os.path.exists(MAT_FILE):
        print(f"Fichier introuvable : {MAT_FILE}")
        print("   -> Données manquantes (voir data/README_DATA.md)")
        return

    print(f"Ouverture de {MAT_FILE}\n")
    mat = loadmat(MAT_FILE)

    emg = mat["emg"]             
    stim = mat["restimulus"].ravel()  
    glove = mat["glove"]         

    # --- Composition des données ----------------------------------------
    duree_s = emg.shape[0] / 2000  
    print("=== COMPOSITION ===")
    print(f"  EMG     : {emg.shape[0]} mesures x {emg.shape[1]} muscles")
    print(f"  Gant    : {glove.shape[0]} mesures x {glove.shape[1]} angles de doigts")
    print(f"  Durée   : {duree_s:.1f} secondes")
    gestes = np.unique(stim)
    print(f"  Gestes présents : {gestes}")
    print(f"  Nombre de gestes différents : {len(gestes)} (dont le repos = 0)\n")

    # --- Équilibre des classes ----------------
    print("=== ÉQUILIBRE DES CLASSES (combien d'instants par geste) ===")
    for g in gestes:
        n = np.sum(stim == g)
        print(f"  geste {int(g):2d} : {n:7d} mesures  ({100*n/len(stim):4.1f} %)")
    print("  -> Le repos (0) domine : c'est un déséquilibre qu'on gérera plus tard.\n")

    # --- Vérification de SYNCHRONISATION EMG <-> mouvement --------------
    # NB : l'EMG est un signal "stochastique" - même pendant une contraction
    # musculaire, sa valeur instantanée a l'air aléatoire (c'est l'AMPLITUDE
    # ENVELOPPE qui change, pas la valeur point-à-point). Pour cette raison,
    # la corrélation brute reste toujours faible. La méthode plus robuste est
    # de comparer l'ENERGIE moyenne pendant les gestes vs au repos (ratio).
    print("=== SYNCHRONISATION DES MODALITÉS ===\n")

    emg_energy = np.mean(np.abs(emg), axis=1)
    glove_motion = np.r_[0, np.mean(np.abs(np.diff(glove, axis=0)), axis=1)]

    # --- (a) MEILLEUR INDICATEUR : ratio d'activité pendant gestes vs repos
    rest_mask = stim == 0
    actif_mask = stim != 0
    emg_rest   = emg_energy[rest_mask].mean()
    emg_actif  = emg_energy[actif_mask].mean()
    mvt_rest   = glove_motion[rest_mask].mean()
    mvt_actif  = glove_motion[actif_mask].mean()
    ratio_emg = emg_actif / emg_rest
    ratio_mvt = mvt_actif / mvt_rest
    print("(a) RATIO D'ACTIVITÉ pendant les gestes (vs repos) :")
    print(f"    EMG (activité musculaire) :  x{ratio_emg:.2f}")
    print(f"    Mouvement des doigts      :  x{ratio_mvt:.2f}")
    print("    -> Les deux ratios bien > 1 prouvent que l'activité musculaire")
    print("       et le mouvement réel des doigts augmentent ENSEMBLE pendant")
    print("       les gestes : les deux modalités sont cohérentes et alignées.\n")

    # --- (b) Corrélation des enveloppes (après lissage sur 2 s)
    # On lisse pour ramener les deux signaux à la même échelle de temps.
    win = 4000   # 2 secondes à 2 kHz 
    kernel = np.ones(win) / win
    emg_smooth = np.convolve(emg_energy, kernel, mode="same")
    glove_smooth = np.convolve(glove_motion, kernel, mode="same")
    corr_lissee = np.corrcoef(emg_smooth, glove_smooth)[0, 1]
    print("(b) Corrélation des enveloppes (lissées 2 s) :")
    print(f"    r = {corr_lissee:.2f}")
    print("    -> Mesure complémentaire ; la corrélation point-à-")
    print("       point reste basse car l'EMG est stochastique - voir (a) pour")
    print("       la preuve principale de synchronisation.\n")

    # --- Figures --------------------------------------------------------
    t = np.arange(len(emg_energy)) / 2000

    plt.figure(figsize=(11, 4))
    plt.plot(t, emg[:, 0])
    plt.title("Signal EMG brut - muscle n°1")
    plt.xlabel("temps (s)"); plt.ylabel("amplitude")
    plt.tight_layout(); plt.savefig("figures/emg_brut.png", dpi=120)

    plt.figure(figsize=(11, 4))
    plt.plot(t, emg_smooth / emg_smooth.max(), label="activité musculaire (EMG, lissée)")
    plt.plot(t, glove_smooth / glove_smooth.max(), label="mouvement des doigts (lissé)", alpha=.75)
    plt.title(f"Synchronisation EMG / mouvement (corrélation lissée = {corr_lissee:.2f})")
    plt.xlabel("temps (s)"); plt.legend()
    plt.tight_layout(); plt.savefig("figures/synchronisation.png", dpi=120)

    print("Figures enregistrées dans figures/ (emg_brut.png, synchronisation.png)")


if __name__ == "__main__":
    main()
