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
        
        # FOCUSED ROI - upper center where traffic lights are
        self.roi = {'y_start': 0.0, 'y_end': 0.6, 'x_start': 0.2, 'x_end': 0.8}
        
        # HSV ranges tuned for Gazebo lighting
        self.red_ranges = {
            'h1': [0, 15], 's1': [150, 255], 'v1': [150, 255],
            'h2': [165, 180], 's2': [150, 255], 'v2': [150, 255]
        }
        self.yellow_ranges = {'h': [18, 35], 's': [150, 255], 'v': [150, 255]}
        self.green_ranges = {'h': [40, 80], 's': [100, 255], 'v': [100, 255]}
        
        # Detection parameters
        self.min_contour_area = 50  # pixels
        self.detection_threshold = 0.002  # 0.2% after contour filtering
        
        # State smoothing
        self.state_buffer = []
        self.buffer_size = 5
        self.last_published = "UNKNOWN"
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.publisher = self.create_publisher(String, '/traffic_light_state', 10)
        self.get_logger().info('Traffic Light Detector V2 STARTED')
        
    def extract_roi(self, img):
        h, w = img.shape[:2]
        y1 = int(h * self.roi['y_start'])
        y2 = int(h * self.roi['y_end'])
        x1 = int(w * self.roi['x_start'])
        x2 = int(w * self.roi['x_end'])
        return img[y1:y2, x1:x2]
    
    def filter_by_shape(self, mask):
        """Keep only circular/elliptical contours (traffic lights)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Traffic lights are circular (circularity near 1.0)
            if circularity > 0.5:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        return filtered_mask
    
    def detect_color(self, hsv, ranges):
        """Detect color with shape filtering"""
        if 'h1' in ranges:  # Red (dual range)
            mask1 = cv2.inRange(hsv, 
                               (ranges['h1'][0], ranges['s1'][0], ranges['v1'][0]),
                               (ranges['h1'][1], ranges['s1'][1], ranges['v1'][1]))
            mask2 = cv2.inRange(hsv,
                               (ranges['h2'][0], ranges['s2'][0], ranges['v2'][0]),
                               (ranges['h2'][1], ranges['s2'][1], ranges['v2'][1]))
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv,
                              (ranges['h'][0], ranges['s'][0], ranges['v'][0]),
                              (ranges['h'][1], ranges['s'][1], ranges['v'][1]))
        
        # Shape filtering
        mask = self.filter_by_shape(mask)
        
        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Extract ROI
            roi = self.extract_roi(cv_image)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Preprocessing
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # Detect each color
            red_mask = self.detect_color(hsv, self.red_ranges)
            yellow_mask = self.detect_color(hsv, self.yellow_ranges)
            green_mask = self.detect_color(hsv, self.green_ranges)
            
            # Calculate ratios
            red_ratio = cv2.countNonZero(red_mask) / total_pixels
            yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
            green_ratio = cv2.countNonZero(green_mask) / total_pixels
            
            # Determine state
            max_ratio = max(red_ratio, yellow_ratio, green_ratio)
            
            if max_ratio < self.detection_threshold:
                detected = "UNKNOWN"
            elif red_ratio == max_ratio and red_ratio > yellow_ratio * 1.5:
                detected = "RED"
            elif yellow_ratio == max_ratio and yellow_ratio > red_ratio * 1.5:
                detected = "YELLOW"
            elif green_ratio == max_ratio:
                detected = "GREEN"
            else:
                detected = "UNKNOWN"
            
            # Buffer-based smoothing
            self.state_buffer.append(detected)
            if len(self.state_buffer) > self.buffer_size:
                self.state_buffer.pop(0)
            
            # Majority vote
            if len(self.state_buffer) == self.buffer_size:
                valid_states = [s for s in self.state_buffer if s != "UNKNOWN"]
                if valid_states:
                    final_state = max(set(valid_states), key=valid_states.count)
                else:
                    final_state = "UNKNOWN"
                
                # Publish if changed
                if final_state != self.last_published:
                    msg_out = String()
                    msg_out.data = final_state
                    self.publisher.publish(msg_out)
                    
                    self.get_logger().info(
                        f'STATE: {final_state} | R:{red_ratio:.4f} Y:{yellow_ratio:.4f} G:{green_ratio:.4f}'
                    )
                    self.last_published = final_state
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')

def main():
    rclpy.init()
    node = TrafficLightDetectorV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()