#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.publisher = self.create_publisher(String, '/traffic_light_state', 10)
        self.get_logger().info('Traffic light detector started')
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            state = self.detect(cv_image)
            
            msg_out = String()
            msg_out.data = state
            self.publisher.publish(msg_out)
            
            if state != "UNKNOWN":
                self.get_logger().info(f'Detected: {state}')
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
        
    def detect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        threshold = 50  # Changed from 500 to 50
        
        if red_pixels > threshold:
            return "RED"
        elif yellow_pixels > threshold:
            return "YELLOW"
        elif green_pixels > threshold:
            return "GREEN"
        
        return "UNKNOWN"
def main():
    rclpy.init()
    node = TrafficLightDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()