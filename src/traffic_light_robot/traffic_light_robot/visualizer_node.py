#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class TrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        
        # Data logging
        self.red_history = deque(maxlen=1000)
        self.green_history = deque(maxlen=1000)
        self.yellow_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        
        self.get_logger().info('Visualizer started - Press Q to quit and generate plot')
        
    def state_callback(self, msg):
        self.current_state = msg.data
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create masks
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Log data
        self.red_history.append(red_pixels)
        self.green_history.append(green_pixels)
        self.yellow_history.append(yellow_pixels)
        self.timestamps.append(self.frame_count)
        self.frame_count += 1
        
        # Visualization
        h, w = cv_image.shape[:2]
        
        # Overlay masks
        red_overlay = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        red_overlay[:,:,1:] = 0
        green_overlay = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        green_overlay[:,:,[0,2]] = 0
        yellow_overlay = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        yellow_overlay[:,:,2] = 0
        
        combined = cv2.addWeighted(cv_image, 0.7, red_overlay, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, green_overlay, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, yellow_overlay, 0.3, 0)
        
        # Info panel
        cv2.rectangle(combined, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # State
        color = (0, 255, 0) if self.current_state == "GREEN" else \
                (0, 255, 255) if self.current_state == "YELLOW" else \
                (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        cv2.putText(combined, f"STATE: {self.current_state}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Pixel counts
        cv2.putText(combined, f"RED: {red_pixels}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(combined, f"YELLOW: {yellow_pixels}", 
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, f"GREEN: {green_pixels}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Threshold line
        threshold = 500
        cv2.line(combined, (250, 50), (250 + threshold//5, 50), (255, 255, 255), 2)
        cv2.putText(combined, f"Threshold: {threshold}", 
                    (250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show
        cv2.imshow("Traffic Light Detection", combined)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_plot()
            raise KeyboardInterrupt
    
    def generate_plot(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data to plot')
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series plot
        ax1.plot(list(self.timestamps), list(self.red_history), 'r-', label='Red', linewidth=1.5)
        ax1.plot(list(self.timestamps), list(self.green_history), 'g-', label='Green', linewidth=1.5)
        ax1.plot(list(self.timestamps), list(self.yellow_history), 'y-', label='Yellow', linewidth=1.5)
        ax1.axhline(y=500, color='white', linestyle='--', label='Threshold', linewidth=2)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Pixel Count')
        ax1.set_title('Traffic Light Color Detection Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1a1a1a')
        
        # Histogram/Distribution
        ax2.hist([list(self.red_history), list(self.green_history), list(self.yellow_history)], 
                 bins=50, label=['Red', 'Green', 'Yellow'], 
                 color=['red', 'green', 'yellow'], alpha=0.7)
        ax2.axvline(x=500, color='white', linestyle='--', label='Threshold', linewidth=2)
        ax2.set_xlabel('Pixel Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Color Pixel Count Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1a1a1a')
        
        # Statistics
        stats_text = f"""Statistics:
Red   - Mean: {np.mean(self.red_history):.0f}, Max: {np.max(self.red_history):.0f}, Std: {np.std(self.red_history):.0f}
Green - Mean: {np.mean(self.green_history):.0f}, Max: {np.max(self.green_history):.0f}, Std: {np.std(self.green_history):.0f}
Yellow- Mean: {np.mean(self.yellow_history):.0f}, Max: {np.max(self.yellow_history):.0f}, Std: {np.std(self.yellow_history):.0f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('traffic_light_analysis.png', dpi=150, facecolor='white')
        self.get_logger().info('Plot saved: traffic_light_analysis.png')
        plt.show()

def main():
    rclpy.init()
    node = TrafficLightVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()