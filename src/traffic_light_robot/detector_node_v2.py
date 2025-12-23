#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightDetectorV2(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_v2')
        self.bridge = CvBridge()
        
        # ULTRA-RELAXED HSV RANGES
        self.red_ranges = {
            'h1': [0, 20], 's1': [10, 255], 'v1': [10, 255],
            'h2': [160, 180], 's2': [10, 255], 'v2': [10, 255]
        }
        self.yellow_ranges = {'h': [15, 50], 's': [10, 255], 'v': [10, 255]}
        self.green_ranges = {'h': [35, 95], 's': [10, 255], 'v': [10, 255]}
        
        # MINIMAL THRESHOLD
        self.detection_threshold = 0.00001  # 0.001%
        
        # STATE FILTERING
        self.last_state = "GREEN"  # Start with GREEN
        self.state_count = 0
        self.state_confirm_threshold = 2
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.publisher = self.create_publisher(String, '/traffic_light_state', 10)
        
        self.get_logger().info('=== DETECTOR V2 STARTED ===')
        self.get_logger().info(f'Detection threshold: {self.detection_threshold}')
        self.get_logger().info(f'Starting state: {self.last_state}')
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # BLUR ONLY
            blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # RED DETECTION (DUAL RANGE)
            r = self.red_ranges
            red1 = cv2.inRange(hsv, (r['h1'][0], r['s1'][0], r['v1'][0]), (r['h1'][1], r['s1'][1], r['v1'][1]))
            red2 = cv2.inRange(hsv, (r['h2'][0], r['s2'][0], r['v2'][0]), (r['h2'][1], r['s2'][1], r['v2'][1]))
            red_mask = red1 | red2
            
            # YELLOW DETECTION
            y = self.yellow_ranges
            yellow_mask = cv2.inRange(hsv, (y['h'][0], y['s'][0], y['v'][0]), (y['h'][1], y['s'][1], y['v'][1]))
            
            # GREEN DETECTION
            g = self.green_ranges
            green_mask = cv2.inRange(hsv, (g['h'][0], g['s'][0], g['v'][0]), (g['h'][1], g['s'][1], g['v'][1]))
            
            # CALCULATE RATIOS
            total = cv_image.shape[0] * cv_image.shape[1]
            red_ratio = cv2.countNonZero(red_mask) / total
            yellow_ratio = cv2.countNonZero(yellow_mask) / total
            green_ratio = cv2.countNonZero(green_mask) / total
            
            # DETERMINE STATE
            max_ratio = max(red_ratio, yellow_ratio, green_ratio)
            
            if max_ratio < self.detection_threshold:
                detected_state = self.last_state  # KEEP PREVIOUS STATE
            elif red_ratio == max_ratio:
                detected_state = "RED"
            elif yellow_ratio == max_ratio:
                detected_state = "YELLOW"
            else:
                detected_state = "GREEN"
            
            # STATE CONFIRMATION
            if detected_state == self.last_state:
                self.state_count += 1
            else:
                self.state_count = 0
            
            # PUBLISH STATE
            if self.state_count >= self.state_confirm_threshold:
                final_state = detected_state
            else:
                final_state = self.last_state
            
            msg_out = String()
            msg_out.data = final_state
            self.publisher.publish(msg_out)
            
            # LOG STATE CHANGES
            if final_state != self.last_state:
                self.get_logger().info(
                    f'STATE CHANGE: {self.last_state} -> {final_state} | '
                    f'R:{red_ratio:.6f} Y:{yellow_ratio:.6f} G:{green_ratio:.6f}'
                )
                self.last_state = final_state
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

def main():
    rclpy.init()
    node = TrafficLightDetectorV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()