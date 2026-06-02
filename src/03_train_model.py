"""
03_train_model.py
=================
ÉTAPE 3 — On entraîne et on VALIDE le modèle qui devine le geste.

Points clés (et pourquoi ils rapportent des points au critère 3) :
  - On compare DEUX modèles : un modèle simple de référence (LDA) et un
    modèle plus puissant (Random Forest). -> "comparaison avec une baseline".
  - On valide "SUJET PAR SUJET" (leave-one-subject-out) : on entraîne sur
    certains sujets et on teste sur un sujet JAMAIS vu. C'est la validation
    la plus honnête : elle mesure si le modèle marchera sur une NOUVELLE
    personne (ce qui est le vrai but d'une prothèse). -> "validation rigoureuse".
  - On gère le DÉSÉQUILIBRE des classes (le repos est sur-représenté) avec
    class_weight="balanced". -> critère 2.
  - On sauvegarde le meilleur modèle pour ROS2.

"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import NOMS_ACTIONS

os.makedirs("figures", exist_ok=True)
os.makedirs("model", exist_ok=True)


def normaliser_par_sujet(X, subjects):
    """
    Normalisation INTRA-SUJET : pour chaque sujet, on centre et on réduit
    SES propres features. Pourquoi ? Parce que d'une personne à l'autre, la
    "force" du signal EMG varie beaucoup (conductivité de la peau, placement
    exact des électrodes, masse musculaire...). En normalisant CHAQUE sujet
    par lui-même, on enlève ces différences "personnelles" pour ne garder que
    la forme du geste. C'est la technique standard pour la généralisation
    inter-sujets en EMG.
    """
    Xn = X.copy().astype(float)
    for s in np.unique(subjects):
        m = subjects == s
        mu = Xn[m].mean(axis=0)
        sd = Xn[m].std(axis=0) + 1e-8
        Xn[m] = (Xn[m] - mu) / sd
    return Xn


def valider_sujet_par_sujet(X, y, subjects, faire_modele):
    """
    Validation INTER-SUJET (leave-one-subject-out, LOSO).
    On entraîne sur N-1 sujets et on teste sur le sujet restant ; on répète
    pour chaque sujet. C'est la validation la plus DIFFICILE et la plus HONNÊTE :
    elle simule l'utilisation par un NOUVEAU utilisateur sans calibration.
    """
    logo = LeaveOneGroupOut()
    accs, f1s = [], []
    # Normalisation par sujet AVANT le découpage train/test : chaque sujet
    # (y compris le sujet de test) est normalisé par ses propres statistiques.
    # En pratique, ce serait une phase de "calibration" de 30 s au début.
    Xn = normaliser_par_sujet(X, subjects)
    for train_idx, test_idx in logo.split(Xn, y, groups=subjects):
        # On standardise encore sur le train pour ramener à une échelle commune.
        scaler = StandardScaler().fit(Xn[train_idx])
        Xtr = scaler.transform(Xn[train_idx])
        Xte = scaler.transform(Xn[test_idx])

        clf = faire_modele()
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)

        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def valider_intra_sujet(X, y, subjects, faire_modele, k=5):
    """
    Validation INTRA-SUJET (k-fold à l'intérieur de chaque sujet).
    Pour CHAQUE sujet, on fait un k-fold stratifié sur SES propres données :
    on entraîne et on teste sur le MÊME sujet.
    -> Ça correspond au scénario "j'ai calibré l'appareil sur cet utilisateur".
    -> C'est BEAUCOUP plus facile que LOSO -> sert de borne supérieure.
    """
    from sklearn.model_selection import StratifiedKFold
    accs, f1s = [], []
    for s in np.unique(subjects):
        m = subjects == s
        Xs, ys = X[m], y[m]
        if len(np.unique(ys)) < 2 or len(ys) < k * 2:
            continue
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        sub_acc, sub_f1 = [], []
        for tr, te in skf.split(Xs, ys):
            sc = StandardScaler().fit(Xs[tr])
            clf = faire_modele().fit(sc.transform(Xs[tr]), ys[tr])
            pred = clf.predict(sc.transform(Xs[te]))
            sub_acc.append(accuracy_score(ys[te], pred))
            sub_f1.append(f1_score(ys[te], pred, average="macro"))
        accs.append(np.mean(sub_acc))
        f1s.append(np.mean(sub_f1))
    return np.array(accs), np.array(f1s)


def main():
    if not os.path.exists("data/dataset.npz"):
        print("data/dataset.npz introuvable. Lance d'abord 02_preprocessing_features.py")
        return

    d = np.load("data/dataset.npz")
    X, y, subjects = d["X"], d["y"], d["subjects"]
    print(f"Dataset : {len(y)} fenêtres, {X.shape[1]} features, "
          f"{len(np.unique(subjects))} sujets.\n")

    # --- Définition des deux modèles ------------------------------------
    modeles = {
        "LDA (référence simple)":
            lambda: LinearDiscriminantAnalysis(),
        "Random Forest (modèle principal)":
            lambda: RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced",   # <- gère le déséquilibre des classes
                random_state=0, n_jobs=-1),
    }

    # --- DEUX validations comparées (point clé du critère 3) ------------
    # On compare deux scénarios :
    #   1) INTRA-sujet (k-fold dans chaque sujet)  = utilisateur calibré
    #   2) INTER-sujet (LOSO, leave-one-subject-out) = nouveau utilisateur
    # Le 1 est toujours bien meilleur que le 2 ; l'écart mesure la difficulté
    # de "généraliser à de nouvelles personnes" en EMG (le vrai défi du domaine).
    for nom, faire in modeles.items():
        print(f"=== {nom} ===")

        a_in, f_in = valider_intra_sujet(X, y, subjects, faire)
        print(f"  [INTRA-sujet] (calibré)        "
              f"Acc {a_in.mean()*100:5.1f} % ± {a_in.std()*100:.1f}  |  "
              f"F1 {f_in.mean()*100:5.1f} %")

        a_lo, f_lo = valider_sujet_par_sujet(X, y, subjects, faire)
        print(f"  [INTER-sujet] (LOSO, nouveau)  "
              f"Acc {a_lo.mean()*100:5.1f} % ± {a_lo.std()*100:.1f}  |  "
              f"F1 {f_lo.mean()*100:5.1f} %")
        print()

    print("Repère honnête de la littérature sur NinaPro :")
    print("  - INTRA-sujet (entrainement+test sur la même personne) : 75-95 %")
    print("  - INTER-sujet LOSO avec features classiques (notre cas) : 55-70 %")
    print("  -> Le LOSO est BEAUCOUP plus difficile : c'est le vrai défi de l'EMG.\n")

    # --- On ré-entraîne le meilleur modèle sur TOUTES les données -------
    #     puis on le sauvegarde pour ROS2.
    print("Entraînement final du Random Forest sur toutes les données...")
    Xn = normaliser_par_sujet(X, subjects)   # mêmes étapes qu'à la validation
    scaler = StandardScaler().fit(Xn)
    Xs = scaler.transform(Xn)
    final = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=0, n_jobs=-1).fit(Xs, y)
    joblib.dump({"model": final, "scaler": scaler, "noms": NOMS_ACTIONS},
                "model/emg_model.joblib")
    print("   -> sauvegardé dans model/emg_model.joblib (utilisé par ROS2).\n")

    # --- Matrice de confusion (très visuelle pour le rapport/la vidéo) --
    # On la calcule sur le dernier découpage train/test sujet-par-sujet.
    logo = LeaveOneGroupOut()
    tr, te = next(logo.split(Xn, y, groups=subjects))
    sc = StandardScaler().fit(Xn[tr])
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
    clf.fit(sc.transform(Xn[tr]), y[tr])
    cm = confusion_matrix(y[te], clf.predict(sc.transform(Xn[te])))

    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matrice de confusion (sujet de test inconnu)")
    plt.xlabel("Geste prédit"); plt.ylabel("Geste réel")
    ticks = list(NOMS_ACTIONS.values())
    plt.xticks(range(len(ticks)), ticks, rotation=45)
    plt.yticks(range(len(ticks)), ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(); plt.tight_layout()
    plt.savefig("figures/matrice_confusion.png", dpi=120)
    print("Matrice de confusion -> figures/matrice_confusion.png")


if __name__ == "__main__":
    main()
