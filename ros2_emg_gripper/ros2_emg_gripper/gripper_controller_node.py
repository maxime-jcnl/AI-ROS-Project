"""
gripper_controller_node.py  (NŒUD ROS2 n°3 : "la main robotique")
================================================================
Ce nœud reçoit la décision (OUVRIR / FERMER / TOURNER / REPOS) et FAIT BOUGER
la pince dans le simulateur (RViz). C'est la "boucle fermée" : un signal
musculaire d'un humain finit par déplacer un robot 
  - chaque action correspond à une position cible des articulations ;
  - on déplace DOUCEMENT les articulations vers la cible (mouvement fluide) ;
  - on publie en continu /joint_states -> RViz affiche la pince qui bouge.

Il écoute :  /gesture
Il publie  :  /joint_states  (lu par RViz / robot_state_publisher)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

# Position cible (en radians/mètres) de chaque articulation, par action.
#   wrist_roll : rotation du poignet ;  finger_* : écartement des doigts.
CIBLES = {
    "REPOS":   {"wrist_roll": 0.0,  "finger_left": 0.01, "finger_right": 0.01},
    "OUVRIR":  {"wrist_roll": 0.0,  "finger_left": 0.04, "finger_right": 0.04},
    "FERMER":  {"wrist_roll": 0.0,  "finger_left": 0.00, "finger_right": 0.00},
    "TOURNER": {"wrist_roll": 1.2,  "finger_left": 0.02, "finger_right": 0.02},
}


class GripperControllerNode(Node):
    def __init__(self):
        super().__init__("gripper_controller")
        self.sub = self.create_subscription(String, "gesture", self.on_gesture, 10)
        self.pub = self.create_publisher(JointState, "joint_states", 10)

        # position actuelle des articulations (point de départ = repos)
        self.pos = dict(CIBLES["REPOS"])
        self.cible = dict(CIBLES["REPOS"])

        # 30 fois par seconde : on rapproche un peu la position de la cible
        self.timer = self.create_timer(1/30, self.bouger)
        self.get_logger().info("Pince prête (en attente de gestes).")

    def on_gesture(self, msg):
        action = msg.data
        if action in CIBLES:
            self.cible = dict(CIBLES[action])
            self.get_logger().info(f"🤖 Nouvelle consigne : {action}")

    def bouger(self):
        # mouvement fluide : on avance de 10 % vers la cible à chaque pas
        for j in self.pos:
            self.pos[j] += 0.1 * (self.cible[j] - self.pos[j])

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["wrist_roll", "finger_left", "finger_right"]
        js.position = [self.pos["wrist_roll"],
                       self.pos["finger_left"],
                       self.pos["finger_right"]]
        self.pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = GripperControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
