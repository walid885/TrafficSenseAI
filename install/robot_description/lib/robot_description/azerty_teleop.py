#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty

class AzertyTeleop(Node):
    def __init__(self):
        super().__init__('azerty_teleop')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.linear_speed = 0.5
        self.angular_speed = 1.0
        
        print("""
╔════════════════════════════════════╗
║   CONTRÔLE ROBOT - AZERTY         ║
╠════════════════════════════════════╣
║  Z : ↑ AVANCER                    ║
║  S : ↓ RECULER                    ║
║  Q : ← GAUCHE (rotation)          ║
║  D : → DROITE (rotation)          ║
║                                    ║
║  Z (maintenu) : + VITESSE         ║
║  S (maintenu) : - VITESSE         ║
║                                    ║
║  ESPACE : ⏹ ARRÊT                 ║
║  ESC : ⏻ QUITTER                  ║
╚════════════════════════════════════╝
Vitesse initiale: 0.5 m/s
        """)
        
        self.settings = termios.tcgetattr(sys.stdin)
        self.last_key = None
        
    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()
                twist = Twist()
                
                if key == 'z':
                    if self.last_key == 'z':
                        self.linear_speed = min(5.0, self.linear_speed + 0.2)
                        print(f'⚡ Vitesse: {self.linear_speed:.1f} m/s')
                    twist.linear.x = self.linear_speed
                    self.publisher.publish(twist)
                    print(f'↑ Avancer ({self.linear_speed:.1f} m/s)')
                    
                elif key == 's':
                    if self.last_key == 's':
                        self.linear_speed = max(0.1, self.linear_speed - 0.2)
                        print(f'⚡ Vitesse: {self.linear_speed:.1f} m/s')
                    twist.linear.x = -self.linear_speed
                    self.publisher.publish(twist)
                    print(f'↓ Reculer ({self.linear_speed:.1f} m/s)')
                    
                elif key == 'q':
                    twist.angular.z = self.angular_speed
                    self.publisher.publish(twist)
                    print('← Rotation gauche')
                    
                elif key == 'd':
                    twist.angular.z = -self.angular_speed
                    self.publisher.publish(twist)
                    print('→ Rotation droite')
                    
                elif key == ' ':
                    self.publisher.publish(Twist())
                    print('⏹ ARRÊT')
                    
                elif key == '\x1b':
                    print('⏻ Arrêt du programme')
                    break
                
                self.last_key = key
                
        except Exception as e:
            print(f'Erreur: {str(e)}')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.publisher.publish(Twist())

def main():
    rclpy.init()
    node = AzertyTeleop()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()