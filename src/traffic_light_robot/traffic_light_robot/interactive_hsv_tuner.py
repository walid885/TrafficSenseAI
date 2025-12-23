#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime

class InteractiveHSVTuner(Node):
    def __init__(self):
        super().__init__('interactive_hsv_tuner')
        self.bridge = CvBridge()
        
        # DEFAULT RANGES
        self.ranges = {
            'RED': {
                'h1': [0, 10], 's1': [100, 255], 'v1': [100, 255],
                'h2': [170, 180], 's2': [100, 255], 'v2': [100, 255]
            },
            'YELLOW': {'h': [20, 35], 's': [100, 255], 'v': [100, 255]},
            'GREEN': {'h': [45, 75], 's': [100, 255], 'v': [100, 255]}
        }
        
        self.current_color = 'RED'
        self.current_image = None
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        cv2.namedWindow("HSV Tuner - Original")
        cv2.namedWindow("HSV Tuner - Mask")
        cv2.namedWindow("HSV Controls")
        
        self.create_trackbars()
        
        self.get_logger().info('='*80)
        self.get_logger().info('INTERACTIVE HSV TUNER')
        self.get_logger().info('='*80)
        self.get_logger().info('Keys:')
        self.get_logger().info('  1 = RED tuning mode')
        self.get_logger().info('  2 = YELLOW tuning mode')
        self.get_logger().info('  3 = GREEN tuning mode')
        self.get_logger().info('  S = Save current parameters')
        self.get_logger().info('  Q = Quit')
        self.get_logger().info('='*80)
    
    def create_trackbars(self):
        cv2.createTrackbar('H_min', 'HSV Controls', 0, 180, lambda x: None)
        cv2.createTrackbar('H_max', 'HSV Controls', 180, 180, lambda x: None)
        cv2.createTrackbar('S_min', 'HSV Controls', 0, 255, lambda x: None)
        cv2.createTrackbar('S_max', 'HSV Controls', 255, 255, lambda x: None)
        cv2.createTrackbar('V_min', 'HSV Controls', 0, 255, lambda x: None)
        cv2.createTrackbar('V_max', 'HSV Controls', 255, 255, lambda x: None)
        
        # RED UPPER RANGE
        cv2.createTrackbar('H2_min', 'HSV Controls', 160, 180, lambda x: None)
        cv2.createTrackbar('H2_max', 'HSV Controls', 180, 180, lambda x: None)
    
    def update_trackbars(self):
        color = self.current_color
        
        if color == 'RED':
            r = self.ranges['RED']
            cv2.setTrackbarPos('H_min', 'HSV Controls', r['h1'][0])
            cv2.setTrackbarPos('H_max', 'HSV Controls', r['h1'][1])
            cv2.setTrackbarPos('S_min', 'HSV Controls', r['s1'][0])
            cv2.setTrackbarPos('S_max', 'HSV Controls', r['s1'][1])
            cv2.setTrackbarPos('V_min', 'HSV Controls', r['v1'][0])
            cv2.setTrackbarPos('V_max', 'HSV Controls', r['v1'][1])
            cv2.setTrackbarPos('H2_min', 'HSV Controls', r['h2'][0])
            cv2.setTrackbarPos('H2_max', 'HSV Controls', r['h2'][1])
        else:
            r = self.ranges[color]
            cv2.setTrackbarPos('H_min', 'HSV Controls', r['h'][0])
            cv2.setTrackbarPos('H_max', 'HSV Controls', r['h'][1])
            cv2.setTrackbarPos('S_min', 'HSV Controls', r['s'][0])
            cv2.setTrackbarPos('S_max', 'HSV Controls', r['s'][1])
            cv2.setTrackbarPos('V_min', 'HSV Controls', r['v'][0])
            cv2.setTrackbarPos('V_max', 'HSV Controls', r['v'][1])
    
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image()
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def process_image(self):
        if self.current_image is None:
            return
        
        img = self.current_image.copy()
        h, w = img.shape[:2]
        
        # ROI
        y1, y2 = int(h * 0.3), int(h * 0.7)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        roi = img[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # GET TRACKBAR VALUES
        h_min = cv2.getTrackbarPos('H_min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Controls')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Controls')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Controls')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Controls')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Controls')
        
        # CREATE MASK
        if self.current_color == 'RED':
            h2_min = cv2.getTrackbarPos('H2_min', 'HSV Controls')
            h2_max = cv2.getTrackbarPos('H2_max', 'HSV Controls')
            mask1 = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            mask2 = cv2.inRange(hsv, (h2_min, s_min, v_min), (h2_max, s_max, v_max))
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        
        # CALCULATE DETECTION RATIO
        ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
        
        # DISPLAY
        display = img.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # TEXT OVERLAY
        text_y = 40
        cv2.putText(display, f"Mode: {self.current_color}", (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        text_y += 50
        cv2.putText(display, f"Detection: {ratio*100:.3f}%", (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        text_y += 40
        cv2.putText(display, f"H: [{h_min}, {h_max}]", (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        text_y += 35
        cv2.putText(display, f"S: [{s_min}, {s_max}]", (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        text_y += 35
        cv2.putText(display, f"V: [{v_min}, {v_max}]", (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # SHOW
        cv2.imshow("HSV Tuner - Original", display)
        
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("HSV Tuner - Mask", mask_display)
        
        # KEYBOARD
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            self.current_color = 'RED'
            self.update_trackbars()
            self.get_logger().info('>>> Switched to RED')
        elif key == ord('2'):
            self.current_color = 'YELLOW'
            self.update_trackbars()
            self.get_logger().info('>>> Switched to YELLOW')
        elif key == ord('3'):
            self.current_color = 'GREEN'
            self.update_trackbars()
            self.get_logger().info('>>> Switched to GREEN')
        elif key == ord('s') or key == ord('S'):
            self.save_parameters()
        elif key == ord('q') or key == ord('Q'):
            self.save_parameters()
            rclpy.shutdown()
    
    def save_parameters(self):
        color = self.current_color
        
        h_min = cv2.getTrackbarPos('H_min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Controls')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Controls')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Controls')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Controls')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Controls')
        
        if color == 'RED':
            h2_min = cv2.getTrackbarPos('H2_min', 'HSV Controls')
            h2_max = cv2.getTrackbarPos('H2_max', 'HSV Controls')
            self.ranges['RED'] = {
                'h1': [h_min, h_max],
                's1': [s_min, s_max],
                'v1': [v_min, v_max],
                'h2': [h2_min, h2_max],
                's2': [s_min, s_max],
                'v2': [v_min, v_max]
            }
        else:
            self.ranges[color] = {
                'h': [h_min, h_max],
                's': [s_min, s_max],
                'v': [v_min, v_max]
            }
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'optimized_ranges': self.ranges
        }
        
        filename = 'hsv_optimized_params.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.get_logger().info(f'Parameters saved to {filename}')
        print("\n" + "="*80)
        print("CURRENT HSV PARAMETERS")
        print("="*80)
        print(json.dumps(self.ranges, indent=2))
        print("="*80 + "\n")

def main():
    rclpy.init()
    node = InteractiveHSVTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()