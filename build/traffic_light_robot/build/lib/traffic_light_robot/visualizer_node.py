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
        
        self.plot_window = 100
        self.plot_red = deque(maxlen=self.plot_window)
        self.plot_green = deque(maxlen=self.plot_window)
        self.plot_yellow = deque(maxlen=self.plot_window)
        self.plot_times = deque(maxlen=self.plot_window)
        
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
        
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (45, 100, 70), (75, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 120, 70), (30, 255, 255))
        
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        red_ratio = (cv2.countNonZero(red_mask) / total_pixels) * 100
        green_ratio = (cv2.countNonZero(green_mask) / total_pixels) * 100
        yellow_ratio = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
        
        ratios = [red_ratio, green_ratio, yellow_ratio]
        sorted_ratios = sorted(ratios, reverse=True)
        confidence = sorted_ratios[0] / (sorted_ratios[1] + 0.01)
        
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        speed_error = abs(self.current_speed - self.target_speed)
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
        
        self.plot_red.append(red_ratio)
        self.plot_green.append(green_ratio)
        self.plot_yellow.append(yellow_ratio)
        self.plot_times.append(elapsed)
        
        self.frame_count += 1
        
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Bigger camera feed
        cam_h, cam_w = 780, 1040
        cam_x, cam_y = 40, 40
        cam_resized = cv2.resize(cv_image, (cam_w, cam_h))
        
        red_overlay = cv2.resize(red_mask, (cam_w, cam_h))
        green_overlay = cv2.resize(green_mask, (cam_w, cam_h))
        yellow_overlay = cv2.resize(yellow_mask, (cam_w, cam_h))
        
        red_bgr = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        red_bgr[:,:,2] = red_overlay
        green_bgr = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        green_bgr[:,:,1] = green_overlay
        yellow_bgr = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        yellow_bgr[:,:,1] = yellow_overlay
        yellow_bgr[:,:,2] = yellow_overlay
        
        cam_with_overlay = cv2.addWeighted(cam_resized, 0.6, red_bgr, 0.4, 0)
        cam_with_overlay = cv2.addWeighted(cam_with_overlay, 1.0, green_bgr, 0.4, 0)
        cam_with_overlay = cv2.addWeighted(cam_with_overlay, 1.0, yellow_bgr, 0.4, 0)
        
        # Tech border
        cv2.rectangle(canvas, (cam_x-5, cam_y-5), (cam_x+cam_w+5, cam_y+cam_h+5), (0, 255, 255), 3)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_with_overlay
        
        cv2.putText(canvas, "LIVE CAMERA FEED + DETECTION", (cam_x, cam_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Timeline plot - bigger and more visible
        plot_x, plot_y = 1120, 40
        plot_w, plot_h = 760, 500
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (20, 20, 30), -1)
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (0, 255, 255), 3)
        
        cv2.putText(canvas, "REAL-TIME COLOR DETECTION (100 Frames)", 
                    (plot_x+100, plot_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        max_val = 1.0
        if len(self.plot_red) > 1:
            plot_data = [
                (list(self.plot_red), (0, 100, 255), "RED"),
                (list(self.plot_green), (0, 255, 100), "GREEN"),
                (list(self.plot_yellow), (0, 255, 255), "YELLOW")
            ]
            
            max_val = max(max(self.plot_red), max(self.plot_green), max(self.plot_yellow), 1.0)
            
            for data, color, name in plot_data:
                points = []
                for i, val in enumerate(data):
                    x = plot_x + 10 + int((i / len(data)) * (plot_w - 20))
                    y = plot_y + plot_h - 10 - int((val / max_val) * (plot_h - 30))
                    points.append((x, y))
                
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(canvas, points[i], points[i+1], color, 3)
        
        # Y-axis labels
        cv2.putText(canvas, "0", (plot_x-35, plot_y+plot_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"{max_val:.0f}", (plot_x-45, plot_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend
        legend_y = plot_y + plot_h + 25
        cv2.rectangle(canvas, (plot_x+50, legend_y), (plot_x+70, legend_y+15), (0, 100, 255), -1)
        cv2.putText(canvas, "Red", (plot_x+80, legend_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+250, legend_y), (plot_x+270, legend_y+15), (0, 255, 255), -1)
        cv2.putText(canvas, "Yellow", (plot_x+280, legend_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+450, legend_y), (plot_x+470, legend_y+15), (0, 255, 100), -1)
        cv2.putText(canvas, "Green", (plot_x+480, legend_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Traffic state - bigger and wider
        state_x = 40
        state_y = cam_y + cam_h + 60
        state_w = 1040
        state_h = 160
        
        state_color = (0, 255, 100) if self.current_state == "GREEN" else \
                      (0, 255, 255) if self.current_state == "YELLOW" else \
                      (0, 100, 255) if self.current_state == "RED" else (128, 128, 128)
        
        cv2.rectangle(canvas, (state_x, state_y), (state_x+state_w, state_y+state_h), (20, 20, 30), -1)
        cv2.rectangle(canvas, (state_x, state_y), (state_x+state_w, state_y+state_h), state_color, 5)
        
        cv2.putText(canvas, "TRAFFIC LIGHT STATUS", (state_x+300, state_y+45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        cv2.putText(canvas, self.current_state, (state_x+320, state_y+120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, state_color, 5)
        
        # Speed control - techy style
        speed_x = 1120
        speed_y = 590
        speed_w = 760
        speed_h = 200
        
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+speed_w, speed_y+speed_h), (20, 20, 30), -1)
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+speed_w, speed_y+speed_h), (0, 255, 255), 3)
        
        cv2.putText(canvas, "ROBOT SPEED CONTROL", (speed_x+200, speed_y+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        
        speed_color = (0, 255, 100) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(canvas, f"Current: {self.current_speed:.3f} m/s", (speed_x+30, speed_y+90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, speed_color, 2)
        cv2.putText(canvas, f"Target:  {self.target_speed:.3f} m/s", (speed_x+30, speed_y+130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"Error:   {speed_error:.3f} m/s", (speed_x+30, speed_y+170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
        
        # Speed bar
        bar_x = speed_x + 420
        bar_w = 310
        bar_h = 120
        bar_y = speed_y + 50
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (30, 30, 40), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100, 100, 120), 2)
        
        max_speed = 0.6
        current_bar = int((self.current_speed / max_speed) * bar_w)
        target_bar = int((self.target_speed / max_speed) * bar_w)
        
        if current_bar > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+current_bar, bar_y+bar_h), speed_color, -1)
        
        target_x = bar_x + target_bar
        cv2.line(canvas, (target_x, bar_y), (target_x, bar_y+bar_h), (255, 0, 100), 5)
        
        # Bottom info bar
        info_y = display_h - 60
        cv2.rectangle(canvas, (0, info_y), (display_w, display_h), (15, 15, 25), -1)
        cv2.rectangle(canvas, (0, info_y), (display_w, info_y+2), (0, 255, 255), 2)
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Confidence: {confidence:.2f} | Press Q for Analytics", 
                    (30, info_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Traffic Light Detection", canvas)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
    
    def generate_analytics(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data')
            return
        
        times = np.array(list(self.timestamps))
        
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual Speed', linewidth=2.5)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target Speed', linewidth=2.5)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2, color='gray')
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Robot Speed Control Performance\nShows how well the robot follows target speed commands', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.98, 'Measures control system responsiveness', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_1_speed_control.png', dpi=200, facecolor='white')
        plt.close()
        
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(times, list(self.red_history), 'r-', label='Red Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.green_history), 'g-', label='Green Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow Light', alpha=0.8, linewidth=2.5)
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Detection Percentage (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Traffic Light Color Detection Over Time\nPercentage of image pixels matching each color', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.02, 0.98, 'Higher peaks indicate stronger color presence', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_2_color_detection.png', dpi=200, facecolor='white')
        plt.close()
        
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(times, list(self.speed_error_history), 'orange', linewidth=2)
        ax3.fill_between(times, 0, list(self.speed_error_history), alpha=0.3, color='orange')
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speed Error (m/s)', fontsize=12, fontweight='bold')
        ax3.set_title('Speed Tracking Error\nAbsolute difference between actual and target speed', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        mean_error = np.mean(self.speed_error_history)
        ax3.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error:.3f} m/s')
        ax3.legend(fontsize=11)
        ax3.text(0.02, 0.98, 'Lower values = better control accuracy', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_3_speed_error.png', dpi=200, facecolor='white')
        plt.close()
        
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(times, list(self.detection_confidence), 'cyan', linewidth=2)
        ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Confidence Threshold')
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence Ratio', fontsize=12, fontweight='bold')
        ax4.set_title('Traffic Light Detection Confidence\nRatio of strongest signal to second strongest', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.text(0.02, 0.98, 'Values >2.0 indicate clear, unambiguous detection', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_4_detection_confidence.png', dpi=200, facecolor='white')
        plt.close()
        
        self.get_logger().info('Analytics saved!')

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