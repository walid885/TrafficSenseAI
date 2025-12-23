#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

class AutoHSVCalibrator(Node):
    def __init__(self):
        super().__init__('auto_hsv_calibrator')
        self.bridge = CvBridge()
        
        self.current_image = None
        self.calibration_data = defaultdict(list)
        self.frame_count = 0
        self.calibration_frames = 100
        self.optimized_ranges = None
        
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.get_logger().info('AUTO CALIBRATOR - Collecting 100 frames...')
    
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if self.optimized_ranges is None:
                self.calibrate()
            else:
                self.detect()
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def calibrate(self):
        img = self.current_image
        h, w = img.shape[:2]
        
        # Full image analysis for traffic lights
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect bright saturated regions
        mask_bright = (hsv[:,:,2] > 100) & (hsv[:,:,1] > 50)
        
        h_vals = hsv[:,:,0][mask_bright]
        s_vals = hsv[:,:,1][mask_bright]
        v_vals = hsv[:,:,2][mask_bright]
        
        if len(h_vals) > 100:
            # RED: 0-15 and 165-180
            red_mask = ((h_vals <= 15) | (h_vals >= 165))
            if np.sum(red_mask) > 50:
                self.calibration_data['RED'].append({
                    'h': h_vals[red_mask],
                    's': s_vals[red_mask],
                    'v': v_vals[red_mask]
                })
            
            # YELLOW: 15-35
            yellow_mask = (h_vals >= 15) & (h_vals <= 35)
            if np.sum(yellow_mask) > 50:
                self.calibration_data['YELLOW'].append({
                    'h': h_vals[yellow_mask],
                    's': s_vals[yellow_mask],
                    'v': v_vals[yellow_mask]
                })
            
            # GREEN: 35-85
            green_mask = (h_vals >= 35) & (h_vals <= 85)
            if np.sum(green_mask) > 50:
                self.calibration_data['GREEN'].append({
                    'h': h_vals[green_mask],
                    's': s_vals[green_mask],
                    'v': v_vals[green_mask]
                })
        
        self.frame_count += 1
        
        # Display progress
        display = img.copy()
        cv2.putText(display, f"Calibrating: {self.frame_count}/{self.calibration_frames}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"RED: {len(self.calibration_data['RED'])} | "
                   f"YEL: {len(self.calibration_data['YELLOW'])} | "
                   f"GRN: {len(self.calibration_data['GREEN'])}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration", display)
        cv2.waitKey(1)
        
        if self.frame_count >= self.calibration_frames:
            self.compute_ranges()
    
    def compute_ranges(self):
        ranges = {}
        
        for color in ['RED', 'YELLOW', 'GREEN']:
            if len(self.calibration_data[color]) == 0:
                continue
            
            all_h = np.concatenate([d['h'] for d in self.calibration_data[color]])
            all_s = np.concatenate([d['s'] for d in self.calibration_data[color]])
            all_v = np.concatenate([d['v'] for d in self.calibration_data[color]])
            
            if color == 'RED':
                # Split into low (0-15) and high (165-180)
                low_h = all_h[all_h <= 90]
                high_h = all_h[all_h > 90]
                
                if len(low_h) > 0:
                    h1_min = max(0, int(np.percentile(low_h, 5)))
                    h1_max = min(20, int(np.percentile(low_h, 95)))
                else:
                    h1_min, h1_max = 0, 10
                
                if len(high_h) > 0:
                    h2_min = max(160, int(np.percentile(high_h, 5)))
                    h2_max = 180
                else:
                    h2_min, h2_max = 170, 180
                
                s_min = max(30, int(np.percentile(all_s, 5)) - 30)
                v_min = max(50, int(np.percentile(all_v, 5)) - 50)
                
                ranges['RED'] = {
                    'h1': [h1_min, h1_max],
                    's1': [s_min, 255],
                    'v1': [v_min, 255],
                    'h2': [h2_min, h2_max],
                    's2': [s_min, 255],
                    'v2': [v_min, 255]
                }
            else:
                h_min = max(0, int(np.percentile(all_h, 5)) - 10)
                h_max = min(180, int(np.percentile(all_h, 95)) + 10)
                s_min = max(30, int(np.percentile(all_s, 5)) - 30)
                v_min = max(50, int(np.percentile(all_v, 5)) - 50)
                
                ranges[color] = {
                    'h': [h_min, h_max],
                    's': [s_min, 255],
                    'v': [v_min, 255]
                }
        
        self.optimized_ranges = ranges
        self.save_calibration()
        self.get_logger().info('CALIBRATION COMPLETE')
        cv2.destroyWindow("Calibration")
    
    def detect(self):
        img = self.current_image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        detections = {}
        masks = {}
        
        for color in ['RED', 'YELLOW', 'GREEN']:
            if color not in self.optimized_ranges:
                continue
            
            r = self.optimized_ranges[color]
            
            if color == 'RED':
                mask1 = cv2.inRange(hsv, (r['h1'][0], r['s1'][0], r['v1'][0]), 
                                   (r['h1'][1], r['s1'][1], r['v1'][1]))
                mask2 = cv2.inRange(hsv, (r['h2'][0], r['s2'][0], r['v2'][0]), 
                                   (r['h2'][1], r['s2'][1], r['v2'][1]))
                mask = mask1 | mask2
            else:
                mask = cv2.inRange(hsv, (r['h'][0], r['s'][0], r['v'][0]), 
                                  (r['h'][1], r['s'][1], r['v'][1]))
            
            ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
            detections[color] = ratio
            masks[color] = mask
        
        detected_color = max(detections, key=detections.get) if detections else None
        confidence = detections.get(detected_color, 0) if detected_color else 0
        
        # Visualization
        display = img.copy()
        
        color_bgr = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
        box_color = color_bgr.get(detected_color, (128, 128, 128))
        
        if detected_color and confidence > 0.001:
            cv2.putText(display, f"{detected_color}: {confidence*100:.3f}%", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)
        else:
            cv2.putText(display, "NO DETECTION", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 3)
        
        y_pos = 80
        for color, ratio in detections.items():
            cv2.putText(display, f"{color}: {ratio*100:.3f}%", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 35
        
        # Show masks
        mask_display = np.zeros_like(img)
        if 'RED' in masks:
            mask_display[:,:,2] = masks['RED']
        if 'YELLOW' in masks:
            mask_display[:,:,1] = cv2.bitwise_or(mask_display[:,:,1], masks['YELLOW'])
            mask_display[:,:,2] = cv2.bitwise_or(mask_display[:,:,2], masks['YELLOW'])
        if 'GREEN' in masks:
            mask_display[:,:,1] = cv2.bitwise_or(mask_display[:,:,1], masks['GREEN'])
        
        combined = cv2.addWeighted(display, 0.7, mask_display, 0.3, 0)
        
        cv2.imshow("Detection", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            rclpy.shutdown()
    
    def save_calibration(self):
        output = {
            'timestamp': datetime.now().isoformat(),
            'method': 'auto_histogram',
            'frames_analyzed': self.frame_count,
            'optimized_ranges': self.optimized_ranges
        }
        
        with open('hsv_optimized_params.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n" + "="*80)
        print("CALIBRATED PARAMETERS")
        print("="*80)
        print(json.dumps(self.optimized_ranges, indent=2))
        print("="*80 + "\n")

def main():
    rclpy.init()
    node = AutoHSVCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()