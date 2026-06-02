"""
emg_classifier_node.py  (NŒUD ROS2 n°2 : "le cerveau")
======================================================
Ce nœud reçoit les fenêtres de signal, devine le geste avec notre modèle de
machine learning, et publie la décision. Il MESURE AUSSI LE TEMPS DE RÉPONSE
(la "latence") : combien de millisecondes entre recevoir le signal et décider.


Il écoute :  /emg_window
Il publie  :  /gesture   (le nom de l'action : OUVRIR, FERMER, ...)

"""

import time
import numpy as np
import joblib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from ros2_emg_gripper.emg_utils import filter_emg, extract_features


class EmgClassifierNode(Node):
    def __init__(self):
        super().__init__("emg_classifier")
        self.declare_parameter("model_path", "model/emg_model.joblib")
        chemin = self.get_parameter("model_path").value

        paquet = joblib.load(chemin)        # contient modèle + scaler + noms
        self.model = paquet["model"]
        self.scaler = paquet["scaler"]
        self.noms = paquet["noms"]

        # --- Calibration "par utilisateur" en ligne ----------------------
        # Pendant l'entraînement, on a normalisé les features de CHAQUE sujet
        # par leurs propres statistiques (moyenne/écart-type). En temps réel,
        # on ne connaît pas ces stats d'avance : on les calcule au fur et à
        # mesure, sur les premières fenêtres reçues (= phase de calibration).
        # Après cela, on les met à jour très lentement pour suivre la dérive.
        self._buf = []                    # features brutes des premières fenêtres
        self._n_calib = 30                # ~6 secondes de calibration au démarrage
        self._mu = None                   # moyenne par feature (apprise)
        self._sd = None                   # écart-type par feature (appris)

        self.sub = self.create_subscription(
            Float32MultiArray, "emg_window", self.on_window, 10)
        self.pub = self.create_publisher(String, "gesture", 10)

        self.latences = []
        self.get_logger().info(f"Classifieur prêt (modèle : {chemin})")
        self.get_logger().info(
            f"⏳ Calibration utilisateur : {self._n_calib} premières fenêtres...")

    def on_window(self, msg):
        t0 = time.perf_counter()

        # 1) on reconstruit la fenêtre (400, 12)
        win = np.array(msg.data, dtype=float).reshape(-1, 12)
        # 2) MÊME nettoyage + MÊMES features que pendant l'entraînement
        win = filter_emg(win)
        feats = extract_features(win)

        # 3) Normalisation "par utilisateur" (cohérent avec l'entraînement)
        if self._mu is None:
            # phase de calibration : on accumule
            self._buf.append(feats)
            if len(self._buf) < self._n_calib:
                return  # on ne décide rien tant qu'on n'est pas calibré
            arr = np.array(self._buf)
            self._mu = arr.mean(axis=0)
            self._sd = arr.std(axis=0) + 1e-8
            self.get_logger().info("✅ Calibration terminée, décisions actives.")
        else:
            # adaptation lente (EMA) pour suivre la dérive du signal
            alpha = 0.01
            self._mu = (1 - alpha) * self._mu + alpha * feats
        feats_n = (feats - self._mu) / self._sd

        # 4) standardisation finale (apprise sur tous les sujets) + prédiction
        feats_n = self.scaler.transform(feats_n.reshape(1, -1))
        action = int(self.model.predict(feats_n)[0])

        latence_ms = (time.perf_counter() - t0) * 1000
        self.latences.append(latence_ms)

        out = String(); out.data = self.noms[action]
        self.pub.publish(out)

        moy = np.mean(self.latences[-50:])
        self.get_logger().info(
            f"🧠 Décision : {self.noms[action]:7s} | latence {latence_ms:5.1f} ms "
            f"(moyenne {moy:4.1f} ms)")


def main(args=None):
    rclpy.init(args=args)
    node = EmgClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.latences:
            node.get_logger().info(
                f"Latence moyenne globale : {np.mean(node.latences):.1f} ms "
                f"(max {np.max(node.latences):.1f} ms)")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
