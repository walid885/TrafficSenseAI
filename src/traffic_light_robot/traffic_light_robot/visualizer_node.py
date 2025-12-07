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
import time

class TrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.0
        
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
        
        cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Traffic Light Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.get_logger().info('Enhanced Visualizer - Press Q for analytics')
        
    def state_callback(self, msg):
        if self.current_state != msg.data and self.current_state != "UNKNOWN":
            self.last_state_change = time.time()
            self.state_change_speed = self.current_speed
        self.current_state = msg.data
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
        if self.last_state_change and self.state_change_speed is not None:
            if abs(self.current_speed - self.target_speed) < 0.05:
                reaction_time = time.time() - self.last_state_change
                if reaction_time < 5.0:
                    self.reaction_times.append(reaction_time)
                self.last_state_change = None
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Normalized HSV ranges
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = red1 | red2
        
        green_mask = cv2.inRange(hsv, (45, 100, 70), (75, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 120, 70), (30, 255, 255))
        
        # Normalized pixel counts (percentage)
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        red_ratio = (cv2.countNonZero(red_mask) / total_pixels) * 100
        green_ratio = (cv2.countNonZero(green_mask) / total_pixels) * 100
        yellow_ratio = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
        
        # Detection confidence
        ratios = [red_ratio, green_ratio, yellow_ratio]
        sorted_ratios = sorted(ratios, reverse=True)
        confidence = sorted_ratios[0] / (sorted_ratios[1] + 0.01)
        
        # Target speed
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        speed_error = abs(self.current_speed - self.target_speed)
        
        # Log data
        elapsed = time.time() - self.start_time
        self.red_history.append(red_ratio)
        self.green_history.append(green_ratio)
        self.yellow_history.append(yellow_ratio)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(confidence)
        self.frame_count += 1
        
        # Fullscreen visualization
        h, w = cv_image.shape[:2]
        display_h, display_w = 1080, 1920
        
        # Create canvas
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Resize camera feed
        cam_h, cam_w = 720, 1280
        cam_resized = cv2.resize(cv_image, (cam_w, cam_h))
        
        # Place camera feed
        canvas[0:cam_h, 0:cam_w] = cam_resized
        
        # Color overlays
        red_overlay = cv2.resize(red_mask, (cam_w, cam_h))
        green_overlay = cv2.resize(green_mask, (cam_w, cam_h))
        yellow_overlay = cv2.resize(yellow_mask, (cam_w, cam_h))
        
        red_overlay_bgr = cv2.cvtColor(red_overlay, cv2.COLOR_GRAY2BGR)
        red_overlay_bgr[:,:,1:] = 0
        green_overlay_bgr = cv2.cvtColor(green_overlay, cv2.COLOR_GRAY2BGR)
        green_overlay_bgr[:,:,[0,2]] = 0
        yellow_overlay_bgr = cv2.cvtColor(yellow_overlay, cv2.COLOR_GRAY2BGR)
        yellow_overlay_bgr[:,:,2] = 0
        
        canvas[0:cam_h, 0:cam_w] = cv2.addWeighted(canvas[0:cam_h, 0:cam_w], 0.7, red_overlay_bgr, 0.3, 0)
        canvas[0:cam_h, 0:cam_w] = cv2.addWeighted(canvas[0:cam_h, 0:cam_w], 1.0, green_overlay_bgr, 0.3, 0)
        canvas[0:cam_h, 0:cam_w] = cv2.addWeighted(canvas[0:cam_h, 0:cam_w], 1.0, yellow_overlay_bgr, 0.3, 0)
        
        # Right panel
        panel_x = cam_w
        panel_w = display_w - cam_w
        
        # State display
        state_y = 100
        color = (0, 255, 0) if self.current_state == "GREEN" else \
                (0, 255, 255) if self.current_state == "YELLOW" else \
                (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        cv2.putText(canvas, "STATE", (panel_x + 50, state_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        cv2.putText(canvas, self.current_state, (panel_x + 50, state_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        
        # Color bars
        bar_y = state_y + 150
        bar_height = 40
        bar_max_width = panel_w - 100
        
        # Red bar
        cv2.putText(canvas, "RED", (panel_x + 20, bar_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        red_bar_w = int((red_ratio / 10.0) * bar_max_width)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + red_bar_w, bar_y + bar_height), (0, 0, 255), -1)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + bar_max_width, bar_y + bar_height), (100, 100, 100), 2)
        cv2.putText(canvas, f"{red_ratio:.2f}%", (panel_x + 120 + bar_max_width + 10, bar_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Yellow bar
        bar_y += 60
        cv2.putText(canvas, "YELLOW", (panel_x + 20, bar_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        yellow_bar_w = int((yellow_ratio / 10.0) * bar_max_width)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + yellow_bar_w, bar_y + bar_height), (0, 255, 255), -1)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + bar_max_width, bar_y + bar_height), (100, 100, 100), 2)
        cv2.putText(canvas, f"{yellow_ratio:.2f}%", (panel_x + 120 + bar_max_width + 10, bar_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Green bar
        bar_y += 60
        cv2.putText(canvas, "GREEN", (panel_x + 20, bar_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        green_bar_w = int((green_ratio / 10.0) * bar_max_width)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + green_bar_w, bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(canvas, (panel_x + 120, bar_y), 
                     (panel_x + 120 + bar_max_width, bar_y + bar_height), (100, 100, 100), 2)
        cv2.putText(canvas, f"{green_ratio:.2f}%", (panel_x + 120 + bar_max_width + 10, bar_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence
        bar_y += 80
        cv2.putText(canvas, f"CONFIDENCE: {confidence:.2f}", (panel_x + 20, bar_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Speed metrics
        speed_y = bar_y + 80
        cv2.putText(canvas, "SPEED CONTROL", (panel_x + 50, speed_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(canvas, f"Current: {self.current_speed:.3f} m/s", (panel_x + 20, speed_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, speed_color, 2)
        cv2.putText(canvas, f"Target:  {self.target_speed:.3f} m/s", (panel_x + 20, speed_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Error:   {speed_error:.3f} m/s", (panel_x + 20, speed_y + 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        
        # Speed variance
        if len(self.speed_history) > 30:
            recent_speeds = list(self.speed_history)[-30:]
            speed_variance = np.std(recent_speeds)
            cv2.putText(canvas, f"Variance: {speed_variance:.4f}", (panel_x + 20, speed_y + 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        # Bottom info
        info_y = display_h - 60
        cv2.rectangle(canvas, (0, info_y), (display_w, display_h), (40, 40, 40), -1)
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Press Q for Analytics", 
                    (20, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Traffic Light Detection", canvas)
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
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual', linewidth=2)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target', linewidth=2)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Speed Control Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(times, list(self.red_history), 'r-', label='Red', alpha=0.8, linewidth=2)
        ax2.plot(times, list(self.green_history), 'g-', label='Green', alpha=0.8, linewidth=2)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Detection (%)')
        ax2.set_title('Traffic Light Color Detection (Normalized)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
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
        
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(times, list(self.speed_error_history), 'orange', linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (m/s)')
        ax4.set_title('Speed Tracking Error')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.plot(times, list(self.detection_confidence), 'cyan', linewidth=1.5)
        ax5.axhline(y=2.0, color='red', linestyle='--', label='Good')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Confidence Ratio')
        ax5.set_title('Detection Confidence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
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
        
        ax7 = fig.add_subplot(gs[3, 1])
        if self.reaction_times:
            ax7.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax7.axvline(np.mean(self.reaction_times), color='red', linestyle='--', linewidth=2)
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Count')
            ax7.set_title(f'Reaction Times (μ={np.mean(self.reaction_times):.2f}s)')
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        stats_text = f"""PERFORMANCE METRICS

Speed Control:
  Mean Speed: {np.mean(self.speed_history):.3f} m/s
  Speed Variance: {np.var(self.speed_history):.4f}
  Mean Error: {np.mean(self.speed_error_history):.3f} m/s
  RMSE: {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f}

Detection (% of image):
  Red    μ={np.mean(self.red_history):.2f}% σ={np.std(self.red_history):.2f}%
  Yellow μ={np.mean(self.yellow_history):.2f}% σ={np.std(self.yellow_history):.2f}%
  Green  μ={np.mean(self.green_history):.2f}% σ={np.std(self.green_history):.2f}%
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