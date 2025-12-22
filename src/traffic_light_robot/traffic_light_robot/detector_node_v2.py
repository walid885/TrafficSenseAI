#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

class TrafficLightDetectorV2(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_v2')
        self.bridge = CvBridge()
        
        self.load_optimized_params()
        
        self.detection_threshold = 0.0001
        self.confidence_history = {'RED': [], 'YELLOW': [], 'GREEN': []}
        self.max_history = 3
        
        self.last_state = "UNKNOWN"
        self.state_persistence = 0
        self.persistence_threshold = 1
        
        # DATA LOGGING
        self.log_data = {
            'session_info': {
                'start_time': datetime.now().isoformat(),
                'hsv_ranges': {
                    'red': self.red_ranges,
                    'yellow': self.yellow_ranges,
                    'green': self.green_ranges
                }
            },
            'frame_data': [],
            'state_changes': []
        }
        
        # History for plotting
        self.red_conf_history = deque(maxlen=2000)
        self.yellow_conf_history = deque(maxlen=2000)
        self.green_conf_history = deque(maxlen=2000)
        self.timestamps = deque(maxlen=2000)
        self.state_history = deque(maxlen=2000)
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_publisher = self.create_publisher(String, '/traffic_light_state', 10)
        
        self.frame_count = 0
        self.start_time = None
        
        self.get_logger().info('Traffic Light Detector v2 with Analytics started')
        self.get_logger().info(f'Detection threshold: {self.detection_threshold}')
        self.get_logger().info('Press Ctrl+C to save logs and generate charts')
        
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
            if self.start_time is None:
                self.start_time = self.get_clock().now()
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
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
            
            # Calculate confidences
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
            
            # LOG DATA
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            
            self.red_conf_history.append(red_smooth)
            self.yellow_conf_history.append(yellow_smooth)
            self.green_conf_history.append(green_smooth)
            self.timestamps.append(elapsed_time)
            self.state_history.append(final_state)
            
            # Log every 10th frame to JSON
            if self.frame_count % 10 == 0:
                frame_log = {
                    'frame': self.frame_count,
                    'timestamp': elapsed_time,
                    'confidences': {
                        'red_raw': float(red_conf),
                        'yellow_raw': float(yellow_conf),
                        'green_raw': float(green_conf),
                        'red_smooth': float(red_smooth),
                        'yellow_smooth': float(yellow_smooth),
                        'green_smooth': float(green_smooth)
                    },
                    'detected_state': final_state
                }
                self.log_data['frame_data'].append(frame_log)
            
            # Log state changes
            if final_state != self.last_state and self.last_state != "UNKNOWN":
                state_change = {
                    'timestamp': elapsed_time,
                    'frame': self.frame_count,
                    'from_state': self.last_state,
                    'to_state': final_state,
                    'confidences': {
                        'red': float(red_smooth),
                        'yellow': float(yellow_smooth),
                        'green': float(green_smooth)
                    }
                }
                self.log_data['state_changes'].append(state_change)
            
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
    
    def save_logs_and_charts(self):
        """Save JSON logs and generate charts on shutdown"""
        
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data to save')
            return
        
        # Add session summary
        self.log_data['session_summary'] = {
            'end_time': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'duration_seconds': float(self.timestamps[-1]) if self.timestamps else 0,
            'state_changes': len(self.log_data['state_changes']),
            'statistics': {
                'red_confidence': {
                    'mean': float(np.mean(self.red_conf_history)),
                    'std': float(np.std(self.red_conf_history)),
                    'max': float(np.max(self.red_conf_history)),
                    'min': float(np.min(self.red_conf_history))
                },
                'yellow_confidence': {
                    'mean': float(np.mean(self.yellow_conf_history)),
                    'std': float(np.std(self.yellow_conf_history)),
                    'max': float(np.max(self.yellow_conf_history)),
                    'min': float(np.min(self.yellow_conf_history))
                },
                'green_confidence': {
                    'mean': float(np.mean(self.green_conf_history)),
                    'std': float(np.std(self.green_conf_history)),
                    'max': float(np.max(self.green_conf_history)),
                    'min': float(np.min(self.green_conf_history))
                }
            }
        }
        
        # Save JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f'detector_v2_log_{timestamp}.json'
        
        try:
            with open(json_filename, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            self.get_logger().info(f'✓ JSON log saved: {json_filename}')
            print(f"\n✓ JSON log saved: {json_filename}")
            print(f"  - {len(self.log_data['frame_data'])} frame samples")
            print(f"  - {len(self.log_data['state_changes'])} state changes")
        except Exception as e:
            self.get_logger().error(f'Failed to save JSON: {e}')
        
        # Generate charts
        self.generate_charts(timestamp)
    
    def generate_charts(self, timestamp):
        """Generate visualization charts"""
        
        times = np.array(list(self.timestamps))
        red_data = np.array(list(self.red_conf_history))
        yellow_data = np.array(list(self.yellow_conf_history))
        green_data = np.array(list(self.green_conf_history))
        
        # Chart 1: Confidence over time
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(times, red_data, 'r-', label='Red', alpha=0.8, linewidth=2)
        ax1.plot(times, yellow_data, 'y-', label='Yellow', alpha=0.8, linewidth=2)
        ax1.plot(times, green_data, 'g-', label='Green', alpha=0.8, linewidth=2)
        ax1.axhline(y=self.detection_threshold, color='black', linestyle='--', 
                   linewidth=1, label=f'Threshold ({self.detection_threshold})')
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Confidence (0-1)', fontsize=12, fontweight='bold')
        ax1.set_title('Traffic Light Detection Confidence Over Time', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        filename1 = f'detector_v2_confidence_{timestamp}.png'
        plt.savefig(filename1, dpi=200, facecolor='white')
        plt.close()
        print(f"✓ Chart saved: {filename1}")
        
        # Chart 2: State timeline
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top: Confidence
        ax2a.plot(times, red_data, 'r-', label='Red', linewidth=2)
        ax2a.plot(times, yellow_data, 'y-', label='Yellow', linewidth=2)
        ax2a.plot(times, green_data, 'g-', label='Green', linewidth=2)
        ax2a.set_ylabel('Confidence', fontsize=11, fontweight='bold')
        ax2a.legend(fontsize=10)
        ax2a.grid(True, alpha=0.3)
        ax2a.set_title('Detection Confidence & State Timeline', fontsize=13, fontweight='bold')
        
        # Bottom: State
        state_numeric = []
        for state in self.state_history:
            if state == "RED":
                state_numeric.append(3)
            elif state == "YELLOW":
                state_numeric.append(2)
            elif state == "GREEN":
                state_numeric.append(1)
            else:
                state_numeric.append(0)
        
        ax2b.plot(times, state_numeric, 'b-', linewidth=2)
        ax2b.set_yticks([0, 1, 2, 3])
        ax2b.set_yticklabels(['UNKNOWN', 'GREEN', 'YELLOW', 'RED'])
        ax2b.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax2b.set_ylabel('State', fontsize=11, fontweight='bold')
        ax2b.grid(True, alpha=0.3)
        ax2b.set_ylim(-0.5, 3.5)
        
        plt.tight_layout()
        filename2 = f'detector_v2_state_timeline_{timestamp}.png'
        plt.savefig(filename2, dpi=200, facecolor='white')
        plt.close()
        print(f"✓ Chart saved: {filename2}")
        
        # Chart 3: Statistics comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        colors = ['red', 'yellow', 'green']
        means = [np.mean(red_data), np.mean(yellow_data), np.mean(green_data)]
        stds = [np.std(red_data), np.std(yellow_data), np.std(green_data)]
        maxs = [np.max(red_data), np.max(yellow_data), np.max(green_data)]
        
        x = np.arange(len(colors))
        width = 0.25
        
        ax3.bar(x - width, means, width, label='Mean', color=['red', 'yellow', 'green'], alpha=0.7)
        ax3.bar(x, stds, width, label='Std Dev', color=['darkred', 'gold', 'darkgreen'], alpha=0.7)
        ax3.bar(x + width, maxs, width, label='Max', color=['lightcoral', 'lightyellow', 'lightgreen'], alpha=0.7)
        
        ax3.set_xlabel('Color Channel', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Confidence Value', fontsize=12, fontweight='bold')
        ax3.set_title('Detection Confidence Statistics by Color', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Red', 'Yellow', 'Green'])
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename3 = f'detector_v2_statistics_{timestamp}.png'
        plt.savefig(filename3, dpi=200, facecolor='white')
        plt.close()
        print(f"✓ Chart saved: {filename3}")
        
        # Chart 4: Heatmap of confidence distribution
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        
        # Create bins
        bins = 50
        hist_red, _ = np.histogram(red_data, bins=bins, range=(0, max(red_data.max(), 0.01)))
        hist_yellow, _ = np.histogram(yellow_data, bins=bins, range=(0, max(yellow_data.max(), 0.01)))
        hist_green, bin_edges = np.histogram(green_data, bins=bins, range=(0, max(green_data.max(), 0.01)))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax4.fill_between(bin_centers, hist_red, alpha=0.5, color='red', label='Red')
        ax4.fill_between(bin_centers, hist_yellow, alpha=0.5, color='yellow', label='Yellow')
        ax4.fill_between(bin_centers, hist_green, alpha=0.5, color='green', label='Green')
        
        ax4.set_xlabel('Confidence Value', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Confidence Distribution Histogram', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename4 = f'detector_v2_distribution_{timestamp}.png'
        plt.savefig(filename4, dpi=200, facecolor='white')
        plt.close()
        print(f"✓ Chart saved: {filename4}")
        
        print(f"\n{'='*60}")
        print("ANALYTICS COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {times[-1]:.2f} seconds")
        print(f"State changes: {len(self.log_data['state_changes'])}")
        print(f"\nConfidence Statistics:")
        print(f"  RED    - Mean: {np.mean(red_data):.6f}, Std: {np.std(red_data):.6f}, Max: {np.max(red_data):.6f}")
        print(f"  YELLOW - Mean: {np.mean(yellow_data):.6f}, Std: {np.std(yellow_data):.6f}, Max: {np.max(yellow_data):.6f}")
        print(f"  GREEN  - Mean: {np.mean(green_data):.6f}, Std: {np.std(green_data):.6f}, Max: {np.max(green_data):.6f}")
        print(f"{'='*60}\n")


def main():
    rclpy.init()
    node = TrafficLightDetectorV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutting down - Saving logs and generating charts...")
        node.save_logs_and_charts()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



