#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from enum import Enum

class State(Enum):
    MOVING = 1
    SLOWING = 2
    STOPPED = 3

class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        
        self.state = State.MOVING
        self.current_light = "UNKNOWN"
        
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Autonomous controller started')
        
    def light_callback(self, msg):
        old_light = self.current_light
        self.current_light = msg.data
        
        if old_light != self.current_light and self.current_light != "UNKNOWN":
            self.get_logger().info(f'Light changed: {old_light} -> {self.current_light}')
        
    def control_loop(self):
        cmd = Twist()
        
        if self.current_light == "RED":
            self.state = State.STOPPED
        elif self.current_light == "YELLOW":
            self.state = State.SLOWING
        elif self.current_light == "GREEN" or self.current_light == "UNKNOWN":
            self.state = State.MOVING
        
        if self.state == State.MOVING:
            cmd.linear.x = 0.5
        elif self.state == State.SLOWING:
            cmd.linear.x = 0.2
        elif self.state == State.STOPPED:
            cmd.linear.x = 0.0
        
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = AutonomousController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    #nAYAHAAHHAHAHA? I AM BACK 