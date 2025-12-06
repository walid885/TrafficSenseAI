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
from scipy import stats
import time

class TrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        # Enhanced data logging
        self.red_history = deque(maxlen=2000)
        self.green_history = deque(maxlen=2000)
        self.yellow_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.target_speed_history = deque(maxlen=2000)
        self.timestamps = deque(maxlen=2000)
        self.state_history = deque(maxlen=2000)
        self.speed_error_history = deque(maxlen=2000)
        self.detection_confidence = deque(maxlen=2000)
        self.reaction_times = []
        self.frame_count = 0
        self.start_time = time.time()
        self.last_state_change = None
        self.state_change_speed = None
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.get_logger().info('Enhanced Visualizer - Press Q for analytics')
        
    def state_callback(self, msg):
        if self.current_state != msg.data and self.current_state != "UNKNOWN":
            self.last_state_change = time.time()
            self.state_change_speed = self.current_speed
        self.current_state = msg.data
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
        # Track reaction time
        if self.last_state_change and self.state_change_speed is not None:
            if abs(self.current_speed - self.target_speed) < 0.05:
                reaction_time = time.time() - self.last_state_change
                if reaction_time < 5.0:
                    self.reaction_times.append(reaction_time)
                self.last_state_change = None
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Color detection
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Detection confidence (ratio of max to second max)
        pixel_counts = [red_pixels, green_pixels, yellow_pixels]
        sorted_counts = sorted(pixel_counts, reverse=True)
        confidence = sorted_counts[0] / (sorted_counts[1] + 1)
        
        # Target speed
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        # Speed tracking error
        speed_error = abs(self.current_speed - self.target_speed)
        
        # Log data
        elapsed = time.time() - self.start_time
        self.red_history.append(red_pixels)
        self.green_history.append(green_pixels)
        self.yellow_history.append(yellow_pixels)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(confidence)
        self.frame_count += 1
        
        # Visualization
        h, w = cv_image.shape[:2]
        
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
        cv2.rectangle(combined, (0, 0), (w, 220), (0, 0, 0), -1)
        
        color = (0, 255, 0) if self.current_state == "GREEN" else \
                (0, 255, 255) if self.current_state == "YELLOW" else \
                (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        cv2.putText(combined, f"STATE: {self.current_state}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(combined, f"R:{red_pixels} G:{green_pixels} Y:{yellow_pixels}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"Confidence: {confidence:.2f}", 
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(combined, f"SPEED: {self.current_speed:.2f} m/s", 
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        cv2.putText(combined, f"TARGET: {self.target_speed:.2f} m/s", 
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"ERROR: {speed_error:.3f} m/s", 
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # Speed variance (rolling std)
        if len(self.speed_history) > 30:
            recent_speeds = list(self.speed_history)[-30:]
            speed_variance = np.std(recent_speeds)
            cv2.putText(combined, f"Variance: {speed_variance:.4f}", 
                        (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Speed bar
        bar_x = w - 150
        bar_y = 30
        bar_width = 120
        bar_height = 140
        
        cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        max_speed = 0.6
        speed_fill_height = int((self.current_speed / max_speed) * bar_height)
        if speed_fill_height > 0:
            cv2.rectangle(combined, 
                         (bar_x + 5, bar_y + bar_height - speed_fill_height), 
                         (bar_x + bar_width - 5, bar_y + bar_height - 5), 
                         speed_color, -1)
        
        target_y = bar_y + bar_height - int((self.target_speed / max_speed) * bar_height)
        cv2.line(combined, (bar_x, target_y), (bar_x + bar_width, target_y), (255, 0, 0), 2)
        
        cv2.imshow("Traffic Light Detection", combined)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
    
    def generate_analytics(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data')
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        times = list(self.timestamps)
        
        # 1. Speed tracking
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual', linewidth=2)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target', linewidth=2)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Speed Control Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Color detection
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(times, list(self.red_history), 'r-', label='Red', alpha=0.8)
        ax2.plot(times, list(self.green_history), 'g-', label='Green', alpha=0.8)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow', alpha=0.8)
        ax2.axhline(y=500, color='white', linestyle='--', label='Threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pixel Count')
        ax2.set_title('Traffic Light Color Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed variance over time
        ax3 = fig.add_subplot(gs[2, 0])
        window = 50
        if len(self.speed_history) > window:
            variances = [np.std(list(self.speed_history)[max(0, i-window):i+1]) 
                        for i in range(len(self.speed_history))]
            ax3.plot(times, variances, 'purple', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Std Dev (m/s)')
        ax3.set_title('Speed Variance (Rolling)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Tracking error
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(times, list(self.speed_error_history), 'orange', linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (m/s)')
        ax4.set_title('Speed Tracking Error')
        ax4.grid(True, alpha=0.3)
        
        # 5. Detection confidence
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.plot(times, list(self.detection_confidence), 'cyan', linewidth=1.5)
        ax5.axhline(y=2.0, color='red', linestyle='--', label='Good')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Confidence Ratio')
        ax5.set_title('Detection Confidence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Speed distribution by state
        ax6 = fig.add_subplot(gs[3, 0])
        states = ['RED', 'YELLOW', 'GREEN']
        speeds_by_state = {s: [] for s in states}
        for state, speed in zip(self.state_history, self.speed_history):
            if state in speeds_by_state:
                speeds_by_state[state].append(speed)
        
        positions = []
        data = []
        labels = []
        for i, state in enumerate(states):
            if speeds_by_state[state]:
                positions.append(i)
                data.append(speeds_by_state[state])
                labels.append(state)
        
        bp = ax6.boxplot(data, positions=positions, labels=labels, patch_artist=True)
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax6.set_ylabel('Speed (m/s)')
        ax6.set_title('Speed Distribution by State')
        ax6.grid(True, alpha=0.3)
        
        # 7. Reaction times
        ax7 = fig.add_subplot(gs[3, 1])
        if self.reaction_times:
            ax7.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax7.axvline(np.mean(self.reaction_times), color='red', linestyle='--', linewidth=2)
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Count')
            ax7.set_title(f'Reaction Times (μ={np.mean(self.reaction_times):.2f}s)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Statistics panel
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        stats_text = f"""PERFORMANCE METRICS
        
Speed Control:
  Mean Speed: {np.mean(self.speed_history):.3f} m/s
  Speed Variance: {np.var(self.speed_history):.4f}
  Mean Error: {np.mean(self.speed_error_history):.3f} m/s
  RMSE: {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f}

Detection:
  Red    μ={np.mean(self.red_history):.0f} σ={np.std(self.red_history):.0f}
  Yellow μ={np.mean(self.yellow_history):.0f} σ={np.std(self.yellow_history):.0f}
  Green  μ={np.mean(self.green_history):.0f} σ={np.std(self.green_history):.0f}
  Avg Confidence: {np.mean(self.detection_confidence):.2f}

Response:
  Reaction Times: {len(self.reaction_times)} measured
  Mean: {np.mean(self.reaction_times) if self.reaction_times else 0:.2f}s
  Std: {np.std(self.reaction_times) if self.reaction_times else 0:.2f}s

Duration: {times[-1]:.1f}s | Frames: {self.frame_count}"""
        
        ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.savefig('traffic_analytics.png', dpi=200, facecolor='white', bbox_inches='tight')
        self.get_logger().info('Analytics saved: traffic_analytics.png')
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