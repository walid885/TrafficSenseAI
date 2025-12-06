#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
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
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        # Data logging
        self.red_history = deque(maxlen=1000)
        self.green_history = deque(maxlen=1000)
        self.yellow_history = deque(maxlen=1000)
        self.speed_history = deque(maxlen=1000)
        self.target_speed_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.get_logger().info('Visualizer started - Press Q to quit and generate plot')
        
    def state_callback(self, msg):
        self.current_state = msg.data
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
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
        
        # Determine target speed based on state
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        # Log data
        self.red_history.append(red_pixels)
        self.green_history.append(green_pixels)
        self.yellow_history.append(yellow_pixels)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
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
        
        # Info panel - EXTENDED
        cv2.rectangle(combined, (0, 0), (w, 180), (0, 0, 0), -1)
        
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
        
        # SPEED DISPLAY
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(combined, f"SPEED: {self.current_speed:.2f} m/s", 
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        cv2.putText(combined, f"TARGET: {self.target_speed:.2f} m/s", 
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Speed bar visualization
        bar_x = w - 150
        bar_y = 30
        bar_width = 120
        bar_height = 140
        
        # Background bar
        cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Current speed bar (max speed = 0.5)
        max_speed = 0.6
        speed_fill_height = int((self.current_speed / max_speed) * bar_height)
        if speed_fill_height > 0:
            cv2.rectangle(combined, 
                         (bar_x + 5, bar_y + bar_height - speed_fill_height), 
                         (bar_x + bar_width - 5, bar_y + bar_height - 5), 
                         speed_color, -1)
        
        # Target speed indicator line
        target_y = bar_y + bar_height - int((self.target_speed / max_speed) * bar_height)
        cv2.line(combined, (bar_x, target_y), (bar_x + bar_width, target_y), (255, 0, 0), 2)
        
        # Labels
        cv2.putText(combined, "SPD", 
                    (bar_x + 35, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        
        # Time series plot - Colors
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
        
        # Time series plot - Speed
        ax2.plot(list(self.timestamps), list(self.speed_history), 'b-', label='Actual Speed', linewidth=2)
        ax2.plot(list(self.timestamps), list(self.target_speed_history), 'r--', label='Target Speed', linewidth=2)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Vehicle Speed Control Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1a1a1a')
        
        # Histogram - Colors
        ax3.hist([list(self.red_history), list(self.green_history), list(self.yellow_history)], 
                 bins=50, label=['Red', 'Green', 'Yellow'], 
                 color=['red', 'green', 'yellow'], alpha=0.7)
        ax3.axvline(x=500, color='white', linestyle='--', label='Threshold', linewidth=2)
        ax3.set_xlabel('Pixel Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Color Pixel Count Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#1a1a1a')
        
        # Histogram - Speed
        ax4.hist(list(self.speed_history), bins=50, color='blue', alpha=0.7, label='Actual Speed')
        ax4.axvline(x=np.mean(self.speed_history), color='red', linestyle='--', label='Mean Speed', linewidth=2)
        ax4.set_xlabel('Speed (m/s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Speed Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('#1a1a1a')
        
        # Statistics
        stats_text = f"""Statistics:
Red    - Mean: {np.mean(self.red_history):.0f}, Max: {np.max(self.red_history):.0f}, Std: {np.std(self.red_history):.0f}
Green  - Mean: {np.mean(self.green_history):.0f}, Max: {np.max(self.green_history):.0f}, Std: {np.std(self.green_history):.0f}
Yellow - Mean: {np.mean(self.yellow_history):.0f}, Max: {np.max(self.yellow_history):.0f}, Std: {np.std(self.yellow_history):.0f}
Speed  - Mean: {np.mean(self.speed_history):.3f}, Max: {np.max(self.speed_history):.3f}, Std: {np.std(self.speed_history):.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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