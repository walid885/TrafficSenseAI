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
import sys

class HSVAutoTuner(Node):
    def __init__(self):
        super().__init__('hsv_auto_tuner')
        self.bridge = CvBridge()
        
        # Data collection
        self.training_data = {'RED': [], 'YELLOW': [], 'GREEN': []}
        self.ground_truth_state = "UNKNOWN"
        self.collection_phase = True
        self.frames_per_state = 50
        self.collected_frames = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
        
        # Diagnostic counters
        self.image_count = 0
        self.state_msg_count = 0
        self.last_diagnostic = self.get_clock().now()
        
        self.red_detected = False
        self.optimization_complete = False
        
        self.best_ranges = {
            'RED': {'h1': [0, 10], 's1': [100, 255], 'v1': [100, 255],
                    'h2': [170, 180], 's2': [100, 255], 'v2': [100, 255]},
            'YELLOW': {'h': [20, 30], 's': [100, 255], 'v': [100, 255]},
            'GREEN': {'h': [45, 75], 's': [100, 255], 'v': [100, 255]}
        }
        
        self.roi = {'y_start': 0.3, 'y_end': 0.7, 'x_start': 0.3, 'x_end': 0.7}
        
        # Subscriptions with diagnostics
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        
        self.result_pub = self.create_publisher(String, '/hsv_tuning_result', 10)
        
        # Diagnostic timer
        self.diagnostic_timer = self.create_timer(2.0, self.print_diagnostics)
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('HSV AUTO-TUNER STARTED')
        self.get_logger().info('Waiting for messages...')
        self.get_logger().info('=' * 80)
        
    def print_diagnostics(self):
        """Print diagnostic information every 2 seconds"""
        if not self.optimization_complete:
            self.get_logger().info(
                f'DIAGNOSTICS | Images: {self.image_count} | '
                f'State msgs: {self.state_msg_count} | '
                f'Current state: {self.ground_truth_state} | '
                f'RED: {self.collected_frames["RED"]}/50 | '
                f'YELLOW: {self.collected_frames["YELLOW"]}/50 | '
                f'GREEN: {self.collected_frames["GREEN"]}/50'
            )
        
    def state_callback(self, msg):
        self.state_msg_count += 1
        self.ground_truth_state = msg.data
        if msg.data == "RED":
            self.red_detected = True
            if self.collected_frames['RED'] == 0:
                self.get_logger().info('>>> RED LIGHT DETECTED - Starting RED collection')
        
    def extract_roi(self, img):
        h, w = img.shape[:2]
        y1 = int(h * self.roi['y_start'])
        y2 = int(h * self.roi['y_end'])
        x1 = int(w * self.roi['x_start'])
        x2 = int(w * self.roi['x_end'])
        return img[y1:y2, x1:x2]
    
    def preprocess(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def image_callback(self, msg):
        self.image_count += 1
        
        if not self.collection_phase or self.optimization_complete:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if self.ground_truth_state in ['RED', 'YELLOW', 'GREEN']:
                state = self.ground_truth_state
                
                if self.collected_frames[state] < self.frames_per_state:
                    roi = self.extract_roi(cv_image)
                    processed = self.preprocess(roi)
                    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
                    
                    self.training_data[state].append(hsv.copy())
                    self.collected_frames[state] += 1
                    
                    if self.collected_frames[state] % 10 == 0:
                        self.get_logger().info(
                            f'>>> Collected {state}: {self.collected_frames[state]}/{self.frames_per_state}')
            
            if (all(count >= self.frames_per_state for count in self.collected_frames.values()) 
                and self.red_detected):
                self.collection_phase = False
                self.get_logger().info('=' * 80)
                self.get_logger().info('COLLECTION COMPLETE - Starting optimization...')
                self.get_logger().info('=' * 80)
                self.start_optimization()
                
        except Exception as e:
            self.get_logger().error(f'Collection error: {str(e)}')
    
    def fitness_function(self, params, color):
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
        
        for state, images in self.training_data.items():
            for hsv_img in images:
                if color == 'RED':
                    mask1 = cv2.inRange(hsv_img, (h1_min, s1_min, v1_min), (h1_max, s1_max, v1_max))
                    mask2 = cv2.inRange(hsv_img, (h2_min, s2_min, v2_min), (h2_max, s2_max, v2_max))
                    mask = mask1 | mask2
                else:
                    mask = cv2.inRange(hsv_img, (h_min, s_min, v_min), (h_max, s_max, v_max))
                
                detection_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
                
                if detection_ratio > 0.005:
                    if state == color:
                        true_positives += 1
                    else:
                        false_positives += 1
        
        fitness = true_positives - 2 * false_positives
        return -fitness
    
    def optimize_color(self, color):
        self.get_logger().info(f'Optimizing {color}...')
        
        if color == 'RED':
            bounds = [
                (0, 10), (5, 15),
                (80, 255), (100, 255),
                (70, 255), (100, 255),
                (165, 175), (170, 180),
                (80, 255), (100, 255),
                (70, 255), (100, 255)
            ]
        elif color == 'YELLOW':
            bounds = [
                (15, 25), (25, 40),
                (80, 255), (100, 255),
                (70, 255), (100, 255)
            ]
        else:
            bounds = [
                (35, 50), (55, 85),
                (80, 255), (100, 255),
                (70, 255), (100, 255)
            ]
        
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
            self.optimization_complete = True
            
            self.get_logger().info('=' * 80)
            self.get_logger().info('OPTIMIZATION COMPLETE - Shutting down node')
            self.get_logger().info('=' * 80)
            rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Optimization error: {str(e)}')
    
    def save_results(self):
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
        print("\n" + "="*80)
        print("OPTIMIZED HSV PARAMETERS")
        print("="*80)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training samples: RED={len(self.training_data['RED'])}, "
              f"YELLOW={len(self.training_data['YELLOW'])}, "
              f"GREEN={len(self.training_data['GREEN'])}")
        print("\n" + "-"*80)
        
        r = self.best_ranges['RED']
        print(f"\nRED:")
        print(f"  Range 1: H=[{r['h1'][0]}, {r['h1'][1]}], S=[{r['s1'][0]}, {r['s1'][1]}], V=[{r['v1'][0]}, {r['v1'][1]}]")
        print(f"  Range 2: H=[{r['h2'][0]}, {r['h2'][1]}], S=[{r['s2'][0]}, {r['s2'][1]}], V=[{r['v2'][0]}, {r['v2'][1]}]")
        
        y = self.best_ranges['YELLOW']
        print(f"\nYELLOW:")
        print(f"  H=[{y['h'][0]}, {y['h'][1]}], S=[{y['s'][0]}, {y['s'][1]}], V=[{y['v'][0]}, {y['v'][1]}]")
        
        g = self.best_ranges['GREEN']
        print(f"\nGREEN:")
        print(f"  H=[{g['h'][0]}, {g['h'][1]}], S=[{g['s'][0]}, {g['s'][1]}], V=[{g['v'][0]}, {g['v'][1]}]")
        
        print("\n" + "="*80)


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