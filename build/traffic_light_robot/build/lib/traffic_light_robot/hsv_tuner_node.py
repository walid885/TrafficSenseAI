#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.optimize import differential_evolution
import json
from datetime import datetime

class HSVAutoTuner(Node):
    """Automated HSV parameter optimizer using differential evolution"""
    
    def __init__(self):
        super().__init__('hsv_auto_tuner')
        self.bridge = CvBridge()
        
        # Data collection
        self.training_data = {'RED': [], 'YELLOW': [], 'GREEN': []}
        self.ground_truth_state = "UNKNOWN"
        self.collection_phase = True
        self.frames_per_state = 50
        self.collected_frames = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
        
        # Best HSV ranges (will be optimized)
        self.best_ranges = {
            'RED': {'h1': [0, 10], 's1': [100, 255], 'v1': [100, 255],
                    'h2': [170, 180], 's2': [100, 255], 'v2': [100, 255]},
            'YELLOW': {'h': [20, 30], 's': [100, 255], 'v': [100, 255]},
            'GREEN': {'h': [45, 75], 's': [100, 255], 'v': [100, 255]}
        }
        
        # ROI parameters
        self.roi = {'y_start': 0.3, 'y_end': 0.7, 'x_start': 0.3, 'x_end': 0.7}
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        
        # Publisher for optimized results
        self.result_pub = self.create_publisher(String, '/hsv_tuning_result', 10)
        
        # Optimization timer (runs after collection phase)
        self.optimization_timer = None
        
        self.get_logger().info('HSV Auto-Tuner started - Collecting training data...')
        
    def state_callback(self, msg):
        """Track ground truth traffic light state"""
        self.ground_truth_state = msg.data
        
    def extract_roi(self, img):
        """Extract region of interest from image"""
        h, w = img.shape[:2]
        y1 = int(h * self.roi['y_start'])
        y2 = int(h * self.roi['y_end'])
        x1 = int(w * self.roi['x_start'])
        x2 = int(w * self.roi['x_end'])
        return img[y1:y2, x1:x2]
    
    def preprocess(self, img):
        """Apply preprocessing for better color detection"""
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # CLAHE to enhance contrast
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def image_callback(self, msg):
        """Collect training images during collection phase"""
        if not self.collection_phase:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Only collect when we have valid ground truth
            if self.ground_truth_state in ['RED', 'YELLOW', 'GREEN']:
                state = self.ground_truth_state
                
                if self.collected_frames[state] < self.frames_per_state:
                    roi = self.extract_roi(cv_image)
                    processed = self.preprocess(roi)
                    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
                    
                    self.training_data[state].append(hsv.copy())
                    self.collected_frames[state] += 1
                    
                    self.get_logger().info(
                        f'Collected {state}: {self.collected_frames[state]}/{self.frames_per_state}')
            
            # Check if collection is complete
            if all(count >= self.frames_per_state for count in self.collected_frames.values()):
                self.collection_phase = False
                self.get_logger().info('Collection complete - Starting optimization...')
                self.start_optimization()
                
        except Exception as e:
            self.get_logger().error(f'Collection error: {str(e)}')
    
    def fitness_function(self, params, color):
        """
        Fitness function for optimization
        Maximizes: true positives - false positives
        """
        if color == 'RED':
            h1_min, h1_max = int(params[0]), int(params[1])
            s1_min, s1_max = int(params[2]), int(params[3])
            v1_min, v1_max = int(params[4]), int(params[5])
            h2_min, h2_max = int(params[6]), int(params[7])
            s2_min, s2_max = int(params[8]), int(params[9])
            v2_min, v2_max = int(params[10]), int(params[11])
        else:
            h_min, h_max = int(params[0]), int(params[1])
            s_min, s_max = int(params[2]), int(params[3])
            v_min, v_max = int(params[4]), int(params[5])
        
        true_positives = 0
        false_positives = 0
        
        # Test on all collected data
        for state, images in self.training_data.items():
            for hsv_img in images:
                if color == 'RED':
                    mask1 = cv2.inRange(hsv_img, (h1_min, s1_min, v1_min), (h1_max, s1_max, v1_max))
                    mask2 = cv2.inRange(hsv_img, (h2_min, s2_min, v2_min), (h2_max, s2_max, v2_max))
                    mask = mask1 | mask2
                else:
                    mask = cv2.inRange(hsv_img, (h_min, s_min, v_min), (h_max, s_max, v_max))
                
                detection_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
                
                # Threshold for detection
                if detection_ratio > 0.005:  # 0.5% of pixels
                    if state == color:
                        true_positives += 1
                    else:
                        false_positives += 1
        
        # Fitness: maximize TP, minimize FP
        fitness = true_positives - 2 * false_positives
        return -fitness  # Negative because scipy minimizes
    
    def optimize_color(self, color):
        """Optimize HSV range for a specific color"""
        self.get_logger().info(f'Optimizing {color}...')
        
        if color == 'RED':
            # Red wraps around HSV, need two ranges
            bounds = [
                (0, 10),      # h1_min, h1_max
                (5, 15),
                (80, 255),    # s1_min, s1_max
                (100, 255),
                (70, 255),    # v1_min, v1_max
                (100, 255),
                (165, 175),   # h2_min, h2_max
                (170, 180),
                (80, 255),    # s2_min, s2_max
                (100, 255),
                (70, 255),    # v2_min, v2_max
                (100, 255)
            ]
        elif color == 'YELLOW':
            bounds = [
                (15, 25),     # h_min, h_max
                (25, 40),
                (80, 255),    # s_min, s_max
                (100, 255),
                (70, 255)     # v_min, v_max
            ]
        else:  # GREEN
            bounds = [
                (35, 50),     # h_min, h_max
                (55, 85),
                (80, 255),    # s_min, s_max
                (100, 255),
                (70, 255)     # v_min, v_max
            ]
        
        # Differential Evolution optimization
        result = differential_evolution(
            lambda x: self.fitness_function(x, color),
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            workers=1,
            updating='deferred'
        )
        
        return result.x
    
    def start_optimization(self):
        """Run optimization for all colors"""
        try:
            optimized_params = {}
            
            for color in ['RED', 'YELLOW', 'GREEN']:
                params = self.optimize_color(color)
                
                if color == 'RED':
                    optimized_params[color] = {
                        'h1': [int(params[0]), int(params[1])],
                        's1': [int(params[2]), int(params[3])],
                        'v1': [int(params[4]), int(params[5])],
                        'h2': [int(params[6]), int(params[7])],
                        's2': [int(params[8]), int(params[9])],
                        'v2': [int(params[10]), int(params[11])]
                    }
                else:
                    optimized_params[color] = {
                        'h': [int(params[0]), int(params[1])],
                        's': [int(params[2]), int(params[3])],
                        'v': [int(params[4]), int(params[5])]
                    }
            
            self.best_ranges = optimized_params
            self.save_results()
            self.print_results()
            
        except Exception as e:
            self.get_logger().error(f'Optimization error: {str(e)}')
    
    def save_results(self):
        """Save optimized parameters to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'optimized_ranges': self.best_ranges,
            'roi': self.roi,
            'training_samples': {k: len(v) for k, v in self.training_data.items()}
        }
        
        filename = 'hsv_optimized_params.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.get_logger().info(f'Results saved to {filename}')
    
    def print_results(self):
        """Print optimized parameters in usable format"""
        print("\n" + "="*80)
        print("OPTIMIZED HSV PARAMETERS")
        print("="*80)
        print("\nCopy to traffic_light_detector.py:\n")
        
        r = self.best_ranges['RED']
        print(f"# Red (dual range)")
        print(f"red1 = cv2.inRange(hsv, ({r['h1'][0]}, {r['s1'][0]}, {r['v1'][0]}), "
              f"({r['h1'][1]}, {r['s1'][1]}, {r['v1'][1]}))")
        print(f"red2 = cv2.inRange(hsv, ({r['h2'][0]}, {r['s2'][0]}, {r['v2'][0]}), "
              f"({r['h2'][1]}, {r['s2'][1]}, {r['v2'][1]}))")
        print("red_mask = red1 | red2\n")
        
        y = self.best_ranges['YELLOW']
        print(f"# Yellow")
        print(f"yellow_mask = cv2.inRange(hsv, ({y['h'][0]}, {y['s'][0]}, {y['v'][0]}), "
              f"({y['h'][1]}, {y['s'][1]}, {y['v'][1]}))\n")
        
        g = self.best_ranges['GREEN']
        print(f"# Green")
        print(f"green_mask = cv2.inRange(hsv, ({g['h'][0]}, {g['s'][0]}, {g['v'][0]}), "
              f"({g['h'][1]}, {g['s'][1]}, {g['v'][1]}))")
        
        print("\n" + "="*80 + "\n")


def main():
    rclpy.init()
    node = HSVAutoTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()