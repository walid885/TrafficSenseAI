#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json

class TrafficLightDetectorV2(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_v2')
        self.bridge = CvBridge()
        
        self.load_optimized_params()
        
        self.detection_threshold = 0.0001  # ULTRA LOW
        self.confidence_history = {'RED': [], 'YELLOW': [], 'GREEN': []}
        self.max_history = 3
        
        self.last_state = "UNKNOWN"
        self.state_persistence = 0
        self.persistence_threshold = 1
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_publisher = self.create_publisher(String, '/traffic_light_state', 10)
        
        self.frame_count = 0
        self.get_logger().info('Traffic Light Detector v2 started')
        self.get_logger().info(f'Detection threshold: {self.detection_threshold}')
        
    def load_optimized_params(self):
        try:
            with open('hsv_optimized_params.json', 'r') as f:
                data = json.load(f)
                params = data['optimized_ranges']
                
                self.red_ranges = params['RED']
                self.yellow_ranges = params['YELLOW']
                self.green_ranges = params['GREEN']
                
                self.get_logger().info('Loaded optimized HSV parameters')
                return
        except:
            pass
        
        # ULTRA RELAXED
        self.red_ranges = {
            'h1': [0, 15], 's1': [20, 255], 'v1': [20, 255],
            'h2': [165, 180], 's2': [20, 255], 'v2': [20, 255]
        }
        self.yellow_ranges = {'h': [10, 45], 's': [20, 255], 'v': [20, 255]}
        self.green_ranges = {'h': [30, 90], 's': [20, 255], 'v': [20, 255]}
        self.get_logger().warn('Using ULTRA-RELAXED defaults')
    
    def preprocess(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        return blurred
    
    def calculate_confidence(self, mask, img_shape):
        total_pixels = img_shape[0] * img_shape[1]
        detected_pixels = cv2.countNonZero(mask)
        return detected_pixels / total_pixels
    
    def smooth_confidence(self, color, new_confidence):
        self.confidence_history[color].append(new_confidence)
        if len(self.confidence_history[color]) > self.max_history:
            self.confidence_history[color].pop(0)
        
        return np.mean(self.confidence_history[color]) if self.confidence_history[color] else 0
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # NO ROI - FULL IMAGE
            processed = self.preprocess(cv_image)
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            
            # Red detection
            r = self.red_ranges
            red1 = cv2.inRange(hsv, 
                              (r['h1'][0], r['s1'][0], r['v1'][0]),
                              (r['h1'][1], r['s1'][1], r['v1'][1]))
            red2 = cv2.inRange(hsv, 
                              (r['h2'][0], r['s2'][0], r['v2'][0]),
                              (r['h2'][1], r['s2'][1], r['v2'][1]))
            red_mask = red1 | red2
            
            # Yellow detection
            y = self.yellow_ranges
            yellow_mask = cv2.inRange(hsv,
                                     (y['h'][0], y['s'][0], y['v'][0]),
                                     (y['h'][1], y['s'][1], y['v'][1]))
            
            # Green detection
            g = self.green_ranges
            green_mask = cv2.inRange(hsv,
                                    (g['h'][0], g['s'][0], g['v'][0]),
                                    (g['h'][1], g['s'][1], g['v'][1]))
            
            # Calculate confidences - FULL IMAGE SHAPE
            red_conf = self.calculate_confidence(red_mask, cv_image.shape)
            yellow_conf = self.calculate_confidence(yellow_mask, cv_image.shape)
            green_conf = self.calculate_confidence(green_mask, cv_image.shape)
            
            # Smooth
            red_smooth = self.smooth_confidence('RED', red_conf)
            yellow_smooth = self.smooth_confidence('YELLOW', yellow_conf)
            green_smooth = self.smooth_confidence('GREEN', green_conf)
            
            # Determine state
            max_conf = max(red_smooth, yellow_smooth, green_smooth)
            
            if max_conf < self.detection_threshold:
                detected_state = "UNKNOWN"
            elif red_smooth == max_conf:
                detected_state = "RED"
            elif yellow_smooth == max_conf:
                detected_state = "YELLOW"
            else:
                detected_state = "GREEN"
            
            # Minimal persistence
            if detected_state == self.last_state:
                self.state_persistence += 1
            else:
                self.state_persistence = 0
            
            if self.state_persistence >= self.persistence_threshold or self.last_state == "UNKNOWN":
                final_state = detected_state
            else:
                final_state = self.last_state
            
            # ALWAYS PUBLISH
            msg_out = String()
            msg_out.data = final_state
            self.state_publisher.publish(msg_out)
            
            # Log every 30 frames
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'State: {final_state} | R:{red_smooth:.6f} Y:{yellow_smooth:.6f} G:{green_smooth:.6f}')
            
            self.last_state = final_state
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()
    node = TrafficLightDetectorV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()