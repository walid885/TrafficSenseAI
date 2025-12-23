#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        
        # STATE
        self.current_light = "GREEN"  # START WITH GREEN
        self.target_speed = 0.5
        
        # SPEED MAPPING
        self.speed_map = {
            "GREEN": 0.5,
            "YELLOW": 0.2,
            "RED": 0.0
        }
        
        # SUBSCRIPTIONS
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # CONTROL TIMER (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('=== CONTROLLER STARTED ===')
        self.get_logger().info(f'Speed map: {self.speed_map}')
        
    def light_callback(self, msg):
        new_light = msg.data
        
        if new_light != self.current_light:
            old_speed = self.target_speed
            self.current_light = new_light
            self.target_speed = self.speed_map.get(new_light, 0.5)
            
            self.get_logger().info(
                f'LIGHT CHANGE: {new_light} | Speed: {old_speed:.2f} -> {self.target_speed:.2f}'
            )
        
    def control_loop(self):
        cmd = Twist()
        cmd.linear.x = self.target_speed
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = AutonomousController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # STOP ROBOT
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()