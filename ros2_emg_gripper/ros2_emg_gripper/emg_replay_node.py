"""
emg_replay_node.py  (NŒUD ROS2 n°1 : "le bras de la personne")
=============================================================
Ce nœud SIMULE le bracelet EMG en temps réel.

Comme on n'a pas de vrai capteur branché, on "rejoue" un enregistrement
NinaPro : on envoie, morceau par morceau (fenêtres de 200 ms), le signal des
12 muscles, comme si quelqu'un faisait des gestes en direct.

Il publie :
  - /emg_window      : la fenêtre de signal brut (4800 nombres = 400 x 12)
  - /emg_true_label  : le geste réellement fait (pour comparer, optionnel)

Pour le lancer seul :  ros2 run ros2_emg_gripper emg_replay
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
from scipy.io import loadmat

from ros2_emg_gripper.emg_utils import (
    GESTE_NINAPRO_VERS_ACTION, NOMS_ACTIONS, WINDOW_SAMPLES, FS)


class EmgReplayNode(Node):
    def __init__(self):
        super().__init__("emg_replay")
        # Paramètre : chemin du fichier .mat à rejouer (à adapter chez toi).
        self.declare_parameter("mat_file", "data/DB2_s1/S1_E1_A1.mat")
        mat_file = self.get_parameter("mat_file").value

        self.pub_emg = self.create_publisher(Float32MultiArray, "emg_window", 10)
        self.pub_lab = self.create_publisher(Int32, "emg_true_label", 10)

        self.windows, self.labels = self._charger(mat_file)
        self.idx = 0

        # On envoie une fenêtre toutes les 0,2 s (rythme "temps réel").
        self.timer = self.create_timer(0.2, self.envoyer_fenetre)
        self.get_logger().info(
            f"Replay prêt : {len(self.windows)} fenêtres chargées depuis {mat_file}")

    def _charger(self, mat_file):
        """Charge le .mat, garde nos 4 gestes, découpe en fenêtres de 200 ms."""
        if not os.path.exists(mat_file):
            self.get_logger().error(f"Fichier introuvable : {mat_file}")
            return [], []
        mat = loadmat(mat_file)
        emg = mat["emg"]
        stim = mat["restimulus"].ravel()
        # on ne garde que les instants de nos 4 gestes choisis
        masque = np.isin(stim, list(GESTE_NINAPRO_VERS_ACTION.keys()))
        emg = emg[masque]
        stim = np.array([GESTE_NINAPRO_VERS_ACTION[s] for s in stim[masque]])

        windows, labels = [], []
        for s in range(0, len(emg) - WINDOW_SAMPLES, WINDOW_SAMPLES):
            win = emg[s:s + WINDOW_SAMPLES]            # (400, 12) signal brut
            lab = np.bincount(stim[s:s + WINDOW_SAMPLES]).argmax()
            windows.append(win.astype(np.float32))
            labels.append(int(lab))
        return windows, labels

    def envoyer_fenetre(self):
        if not self.windows:
            return
        win = self.windows[self.idx]
        lab = self.labels[self.idx]

        msg = Float32MultiArray()
        msg.data = win.flatten().tolist()   # 400x12 -> 4800 nombres
        self.pub_emg.publish(msg)

        lmsg = Int32(); lmsg.data = lab
        self.pub_lab.publish(lmsg)

        self.get_logger().info(f"➡️  Envoi fenêtre {self.idx} | vrai geste : {NOMS_ACTIONS[lab]}")
        self.idx = (self.idx + 1) % len(self.windows)  # on boucle


def main(args=None):
    rclpy.init(args=args)
    node = EmgReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
