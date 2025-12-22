#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os

class TrafficLightDetectorV2(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_v2')
        self.bridge = CvBridge()
        
        # Load optimized parameters if available
        self.load_optimized_params()
        
        # ROI parameters
        self.roi = {'y_start': 0.3, 'y_end': 0.7, 'x_start': 0.3, 'x_end': 0.7}
        
        # Adaptive thresholding
        self.detection_threshold = 0.003  # 0.3% of ROI pixels
        self.confidence_history = {'RED': [], 'YELLOW': [], 'GREEN': []}
        self.max_history = 10
        
        # State filtering
        self.last_state = "UNKNOWN"
        self.state_persistence = 0
        self.persistence_threshold = 3
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)

        
        self.publisher = self.create_publisher(String, '/traffic_light_state', 10)
        self.get_logger().info('Optimized Traffic Light Detector v2 started')
        self.ground_truth_publisher = self.create_publisher(String, '/traffic_light_state', 10)

    def load_optimized_params(self):
        """Load optimized HSV parameters from JSON if available"""
        try:
            with open('hsv_optimized_params.json', 'r') as f:
                data = json.load(f)
                params = data['optimized_ranges']
                
                self.red_ranges = params['RED']
                self.yellow_ranges = params['YELLOW']
                self.green_ranges = params['GREEN']
                
                self.get_logger().info('Loaded optimized HSV parameters')
                return
        except FileNotFoundError:
            self.get_logger().warn('No optimized params found, using defaults')
        except Exception as e:
            self.get_logger().error(f'Error loading params: {e}')
        
        # Fallback to enhanced default parameters
        self.red_ranges = {
            'h1': [0, 10], 's1': [100, 255], 'v1': [100, 255],
            'h2': [170, 180], 's2': [100, 255], 'v2': [100, 255]
        }
        self.yellow_ranges = {'h': [20, 32], 's': [100, 255], 'v': [100, 255]}
        self.green_ranges = {'h': [45, 75], 's': [100, 255], 'v': [100, 255]}
    
    def extract_roi(self, img):
        """Extract region of interest"""
        h, w = img.shape[:2]
        y1 = int(h * self.roi['y_start'])
        y2 = int(h * self.roi['y_end'])
        x1 = int(w * self.roi['x_start'])
        x2 = int(w * self.roi['x_end'])
        return img[y1:y2, x1:x2]
    
    def preprocess(self, img):
        """Preprocessing pipeline for robust detection"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_with_morphology(self, mask):
        """Apply morphological operations to clean mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def calculate_confidence(self, mask, roi_shape):
        """Calculate detection confidence with smoothing"""
        total_pixels = roi_shape[0] * roi_shape[1]
        detected_pixels = cv2.countNonZero(mask)
        ratio = detected_pixels / total_pixels
        return ratio
    
    def smooth_confidence(self, color, new_confidence):
        """Smooth confidence over time"""
        self.confidence_history[color].append(new_confidence)
        if len(self.confidence_history[color]) > self.max_history:
            self.confidence_history[color].pop(0)
        
        return np.mean(self.confidence_history[color]) if self.confidence_history[color] else 0
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Extract ROI and preprocess
            roi = self.extract_roi(cv_image)
            processed = self.preprocess(roi)
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            
            # Red detection (dual range)
            r = self.red_ranges
            red1 = cv2.inRange(hsv, 
                              (r['h1'][0], r['s1'][0], r['v1'][0]),
                              (r['h1'][1], r['s1'][1], r['v1'][1]))
            red2 = cv2.inRange(hsv, 
                              (r['h2'][0], r['s2'][0], r['v2'][0]),
                              (r['h2'][1], r['s2'][1], r['v2'][1]))
            red_mask = self.detect_with_morphology(red1 | red2)
            
            # Yellow detection
            y = self.yellow_ranges
            yellow_mask = cv2.inRange(hsv,
                                     (y['h'][0], y['s'][0], y['v'][0]),
                                     (y['h'][1], y['s'][1], y['v'][1]))
            yellow_mask = self.detect_with_morphology(yellow_mask)
            
            # Green detection
            g = self.green_ranges
            green_mask = cv2.inRange(hsv,
                                    (g['h'][0], g['s'][0], g['v'][0]),
                                    (g['h'][1], g['s'][1], g['v'][1]))
            green_mask = self.detect_with_morphology(green_mask)
            
            # Calculate and smooth confidences
            red_conf = self.smooth_confidence('RED', 
                                            self.calculate_confidence(red_mask, roi.shape))
            yellow_conf = self.smooth_confidence('YELLOW',
                                                self.calculate_confidence(yellow_mask, roi.shape))
            green_conf = self.smooth_confidence('GREEN',
                                               self.calculate_confidence(green_mask, roi.shape))
            
            # Determine state with hysteresis
            max_conf = max(red_conf, yellow_conf, green_conf)


            
            if max_conf < self.detection_threshold:
                detected_state = "UNKNOWN"
            elif red_conf == max_conf:
                detected_state = "RED"
            elif yellow_conf == max_conf:
                detected_state = "YELLOW"
            else:
                detected_state = "GREEN"
            
            # State persistence filtering
            if detected_state == self.last_state or detected_state == "UNKNOWN":
                self.state_persistence += 1
            else:
                self.state_persistence = 0
            
            if self.state_persistence >= self.persistence_threshold:
                final_state = detected_state
            else:
                final_state = self.last_state if self.last_state != "UNKNOWN" else detected_state
            
            state_msg = String()
            state_msg.data = final_state
            self.state_publisher.publish(state_msg)

            # Publish
            msg_out = String()
            msg_out.data = final_state
            self.publisher.publish(msg_out)
            self.ground_truth_publisher.publish(msg_out)  # ADD THIS

            if final_state != self.last_state and final_state != "UNKNOWN":
                self.get_logger().info(
                    f'State: {final_state} | R:{red_conf:.4f} Y:{yellow_conf:.4f} G:{green_conf:.4f}')
                self.last_state = final_state
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')


def main():
    rclpy.init()
    node = TrafficLightDetectorV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
