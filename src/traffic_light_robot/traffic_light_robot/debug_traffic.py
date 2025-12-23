#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class TrafficDebugger(Node):
    def __init__(self):
        super().__init__('traffic_debugger')
        
        self.light_state = "UNKNOWN"
        self.cmd_vel = 0.0
        
        self.create_subscription(String, '/traffic_light_state', self.light_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        
        self.create_timer(1.0, self.print_status)
        
        self.get_logger().info('=== DEBUG NODE STARTED ===')
        
    def light_cb(self, msg):
        self.light_state = msg.data
        
    def cmd_cb(self, msg):
        self.cmd_vel = msg.linear.x
        
    def print_status(self):
        self.get_logger().info(
            f'LIGHT: {self.light_state:8s} | CMD_VEL: {self.cmd_vel:.3f} m/s'
        )

def main():
    rclpy.init()
    node = TrafficDebugger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()