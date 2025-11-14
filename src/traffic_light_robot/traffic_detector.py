#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.state_pub = self.create_publisher(String, '/traffic_light_state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Traffic Light Detector Node Started')
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_traffic_light(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def detect_traffic_light(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red detection (two ranges)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Green detection
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Count pixels
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        # Determine state
        threshold = 500
        if red_pixels > threshold:
            self.current_state = "RED"
        elif green_pixels > threshold:
            self.current_state = "GREEN"
        elif yellow_pixels > threshold:
            self.current_state = "YELLOW"
        else:
            self.current_state = "NONE"
        
        # Publish state
        state_msg = String()
        state_msg.data = self.current_state
        self.state_pub.publish(state_msg)
        
        self.get_logger().info(f'Traffic Light State: {self.current_state}')
    
    def control_loop(self):
        twist = Twist()
        
        if self.current_state == "RED":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif self.current_state == "YELLOW":
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        elif self.current_state == "GREEN":
            twist.linear.x = 0.5
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.3
            twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
