
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
import json
from datetime import datetime

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
        
        # JSON logging data structures
        self.json_log = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "node_name": self.get_name()
            },
            "frame_data": [],
            "state_changes": [],
            "reaction_events": []
        }
        
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
        
        self.get_logger().info('Enhanced Visualizer with JSON Logging - Press Q for analytics')
        
    def state_callback(self, msg):
        if self.current_state != msg.data and self.current_state != "UNKNOWN":
            elapsed = time.time() - self.start_time
            self.last_state_change = time.time()
            self.state_change_speed = self.current_speed
            
            # Log state change to JSON
            state_change_event = {
                "timestamp": elapsed,
                "frame": self.frame_count,
                "previous_state": self.current_state,
                "new_state": msg.data,
                "speed_at_change": self.current_speed,
                "target_speed": self.target_speed
            }
            self.json_log["state_changes"].append(state_change_event)
            
        self.current_state = msg.data
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
        if self.last_state_change and self.state_change_speed is not None:
            if abs(self.current_speed - self.target_speed) < 0.05:
                reaction_time = time.time() - self.last_state_change
                if reaction_time < 5.0:
                    self.reaction_times.append(reaction_time)
                    
                    # Log reaction event to JSON
                    reaction_event = {
                        "timestamp": time.time() - self.start_time,
                        "reaction_time": reaction_time,
                        "initial_speed": self.state_change_speed,
                        "final_speed": self.current_speed,
                        "target_speed": self.target_speed,
                        "state": self.current_state
                    }
                    self.json_log["reaction_events"].append(reaction_event)
                    
                self.last_state_change = None
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Use detector_node_v2 ranges (relaxed thresholds)
        red1 = cv2.inRange(hsv, (0, 20, 20), (15, 255, 255))
        red2 = cv2.inRange(hsv, (165, 20, 20), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (30, 20, 20), (90, 255, 255))
        yellow_mask = cv2.inRange(hsv, (10, 20, 20), (45, 255, 255))
        
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        red_ratio = (cv2.countNonZero(red_mask) / total_pixels)
        green_ratio = (cv2.countNonZero(green_mask) / total_pixels)
        yellow_ratio = (cv2.countNonZero(yellow_mask) / total_pixels)
        
        # Match detector_v2 confidence calculation
        ratios = [red_ratio, green_ratio, yellow_ratio]
        max_ratio = max(ratios)
        
        # Store actual ratio values (0-1 scale)
        confidence = max_ratio
        
        dominant_color = "RED" if red_ratio == max_ratio else \
                        "GREEN" if green_ratio == max_ratio else \
                        "YELLOW" if yellow_ratio == max_ratio else "NONE"
        
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
        
        if self.frame_count % 5 == 0:
            frame_data = {
                "frame": self.frame_count,
                "timestamp": elapsed,
                "color_detection": {
                    "red_ratio": round(red_ratio, 6),
                    "green_ratio": round(green_ratio, 6),
                    "yellow_ratio": round(yellow_ratio, 6),
                    "dominant_color": dominant_color,
                    "confidence": round(confidence, 6)
                },
                "speed_control": {
                    "current_speed": round(self.current_speed, 4),
                    "target_speed": round(self.target_speed, 4),
                    "speed_error": round(speed_error, 4)
                },
                "traffic_state": self.current_state
            }
            self.json_log["frame_data"].append(frame_data)
        
        # Store as ratio (0-1) for accurate plotting
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
        
        # LARGER DISPLAY
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # BIGGER CAMERA VIEW
        cam_h, cam_w = 750, 1000
        cam_x, cam_y = 30, 30
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
        
        cv2.rectangle(canvas, (cam_x-5, cam_y-5), (cam_x+cam_w+5, cam_y+cam_h+5), (255, 255, 255), 3)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_with_overlay
        
        cv2.putText(canvas, "CAMERA FEED + COLOR DETECTION", (cam_x, cam_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # PLOT AREA
        plot_x, plot_y = 1070, 30
        plot_w, plot_h = 800, 400
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (255, 255, 255), 2)
        
        cv2.putText(canvas, "CONFIDENCE TIMELINE (0-1 scale)", 
                    (plot_x+10, plot_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        max_val = 0.02  # Show 0 to 0.02 range for better visibility
        if len(self.plot_red) > 1:
            plot_data = [
                (list(self.plot_red), (0, 0, 255)),
                (list(self.plot_green), (0, 255, 0)),
                (list(self.plot_yellow), (0, 255, 255))
            ]
            
            current_max = max(max(self.plot_red), max(self.plot_green), max(self.plot_yellow))
            if current_max > max_val:
                max_val = current_max * 1.2
            
            for data, color in plot_data:
                points = []
                for i, val in enumerate(data):
                    x = plot_x + int((i / len(data)) * plot_w)
                    y = plot_y + plot_h - int((val / max_val) * (plot_h - 20))
                    points.append((x, y))
                
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(canvas, points[i], points[i+1], color, 2)
        
        cv2.putText(canvas, "0", (plot_x-25, plot_y+plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"{max_val:.4f}", (plot_x-50, plot_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        legend_y = plot_y + plot_h + 20
        cv2.rectangle(canvas, (plot_x, legend_y), (plot_x+20, legend_y+20), (0, 0, 255), -1)
        cv2.putText(canvas, "Red", (plot_x+30, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+120, legend_y), (plot_x+140, legend_y+20), (0, 255, 255), -1)
        cv2.putText(canvas, "Yellow", (plot_x+150, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+260, legend_y), (plot_x+280, legend_y+20), (0, 255, 0), -1)
        cv2.putText(canvas, "Green", (plot_x+290, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        state_x = 30
        state_y = cam_y + cam_h + 50
        
        state_color = (0, 255, 0) if self.current_state == "GREEN" else \
                    (0, 255, 255) if self.current_state == "YELLOW" else \
                    (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        
        cv2.rectangle(canvas, (state_x, state_y), (state_x+480, state_y+150), (50, 50, 50), -1)
        cv2.rectangle(canvas, (state_x, state_y), (state_x+480, state_y+150), state_color, 4)
        
        cv2.putText(canvas, "TRAFFIC STATE", (state_x+120, state_y+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(canvas, self.current_state, (state_x+90, state_y+115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, state_color, 4)
        
        metrics_x = 550
        metrics_y = state_y
        
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+480, metrics_y+150), (50, 50, 50), -1)
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+480, metrics_y+150), (255, 255, 255), 2)
        
        cv2.putText(canvas, "DETECTION CONFIDENCE", (metrics_x+70, metrics_y+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, f"R: {red_ratio:.6f}", (metrics_x+30, metrics_y+75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(canvas, f"Y: {yellow_ratio:.6f}", (metrics_x+30, metrics_y+105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"G: {green_ratio:.6f}", (metrics_x+30, metrics_y+135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        speed_x = 1070
        speed_y = 480
        
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+800, speed_y+200), (50, 50, 50), -1)
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+800, speed_y+200), (255, 255, 255), 2)
        
        cv2.putText(canvas, "ROBOT SPEED CONTROL", (speed_x+220, speed_y+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(canvas, f"Current: {self.current_speed:.3f} m/s", (speed_x+30, speed_y+90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, speed_color, 2)
        cv2.putText(canvas, f"Target:  {self.target_speed:.3f} m/s", (speed_x+30, speed_y+135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Error:   {speed_error:.3f} m/s", (speed_x+30, speed_y+180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)
        
        bar_x = speed_x + 450
        bar_w = 300
        bar_h = 120
        bar_y = speed_y + 40
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100, 100, 100), 2)
        
        max_speed = 0.6
        current_bar = int((self.current_speed / max_speed) * bar_w)
        target_bar = int((self.target_speed / max_speed) * bar_w)
        
        if current_bar > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+current_bar, bar_y+bar_h), speed_color, -1)
        
        target_x = bar_x + target_bar
        cv2.line(canvas, (target_x, bar_y), (target_x, bar_y+bar_h), (255, 0, 0), 4)
        
        info_y = display_h - 50
        cv2.rectangle(canvas, (0, info_y), (display_w, display_h), (40, 40, 40), -1)
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Max Conf: {confidence:.6f} | Press Q for Analytics", 
                    (30, info_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Traffic Light Detection", canvas)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
        
    def save_json_log(self, filename='traffic_light_detection_log.json'):
        """Save comprehensive JSON log of all detection data"""
        
        # Add session summary
        self.json_log["session_summary"] = {
            "end_time": datetime.now().isoformat(),
            "total_duration": float(self.timestamps[-1]) if self.timestamps else 0,
            "total_frames": self.frame_count,
            "total_state_changes": len(self.json_log["state_changes"]),
            "total_reaction_events": len(self.json_log["reaction_events"]),
            "statistics": {
                "speed": {
                    "mean": float(np.mean(self.speed_history)) if self.speed_history else 0,
                    "std": float(np.std(self.speed_history)) if self.speed_history else 0,
                    "max": float(np.max(self.speed_history)) if self.speed_history else 0,
                    "min": float(np.min(self.speed_history)) if self.speed_history else 0
                },
                "speed_error": {
                    "mean": float(np.mean(self.speed_error_history)) if self.speed_error_history else 0,
                    "rmse": float(np.sqrt(np.mean(np.array(self.speed_error_history)**2))) if self.speed_error_history else 0
                },
                "color_detection": {
                    "red": {
                        "mean": float(np.mean(self.red_history)) if self.red_history else 0,
                        "max": float(np.max(self.red_history)) if self.red_history else 0
                    },
                    "green": {
                        "mean": float(np.mean(self.green_history)) if self.green_history else 0,
                        "max": float(np.max(self.green_history)) if self.green_history else 0
                    },
                    "yellow": {
                        "mean": float(np.mean(self.yellow_history)) if self.yellow_history else 0,
                        "max": float(np.max(self.yellow_history)) if self.yellow_history else 0
                    }
                },
                "confidence": {
                    "mean": float(np.mean(self.detection_confidence)) if self.detection_confidence else 0,
                    "std": float(np.std(self.detection_confidence)) if self.detection_confidence else 0
                },
                "reaction_time": {
                    "mean": float(np.mean(self.reaction_times)) if self.reaction_times else 0,
                    "std": float(np.std(self.reaction_times)) if self.reaction_times else 0,
                    "count": len(self.reaction_times)
                }
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.json_log, f, indent=2)
            self.get_logger().info(f'JSON log saved to {filename}')
            print(f"\n✓ Comprehensive JSON log saved: {filename}")
            print(f"  - {len(self.json_log['frame_data'])} frame samples")
            print(f"  - {len(self.json_log['state_changes'])} state changes")
            print(f"  - {len(self.json_log['reaction_events'])} reaction events")
        except Exception as e:
            self.get_logger().error(f'Failed to save JSON log: {e}')
    
    def generate_analytics(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data')
            return
        
        # Save JSON log first
        self.save_json_log('traffic_light_detection_log.json')
        
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
        ax4.axhline(y=20.0, color='red', linestyle='--', linewidth=2, label='Good Confidence (20%)')
        ax4.axhline(y=50.0, color='green', linestyle='--', linewidth=2, label='Excellent (50%)')
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Traffic Light Detection Confidence\nSignal-to-noise ratio (0-100% scale)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.text(0.02, 0.98, 'Higher = clearer detection. >20% is good, >50% is excellent', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_4_detection_confidence.png', dpi=200, facecolor='white')
        plt.close()
        
        fig5, ax5 = plt.subplots(figsize=(14, 6))
        window = 50
        if len(self.speed_history) > window:
            variances = [np.std(list(self.speed_history)[max(0, i-window):i+1]) 
                        for i in range(len(self.speed_history))]
            ax5.plot(times, variances, 'purple', linewidth=2)
            ax5.fill_between(times, 0, variances, alpha=0.3, color='purple')
        ax5.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Standard Deviation (m/s)', fontsize=12, fontweight='bold')
        ax5.set_title('Speed Stability (Rolling Window)\nMeasures speed fluctuations over 50-frame windows', 
                     fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True, alpha=0.3)
        ax5.text(0.02, 0.98, 'Lower variance = smoother, more stable motion', 
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_5_speed_variance.png', dpi=200, facecolor='white')
        plt.close()
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
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
        
        if data:
            bp = ax6.boxplot(data, positions=positions, labels=labels, patch_artist=True, widths=0.6)
            colors = ['red', 'yellow', 'green']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax6.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Traffic Light State', fontsize=12, fontweight='bold')
        ax6.set_title('Speed Distribution by Traffic Light State\nShows typical speed ranges for each light state', 
                     fontsize=14, fontweight='bold', pad=20)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.text(0.02, 0.98, 'Red should be near 0, Green higher', 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('analytics_6_speed_by_state.png', dpi=200, facecolor='white')
        plt.close()
        
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        if self.reaction_times:
            ax7.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            mean_rt = np.mean(self.reaction_times)
            ax7.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}s')
            ax7.set_xlabel('Reaction Time (seconds)', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax7.set_title(f'Robot Reaction Time Distribution\nTime taken to reach target speed after light change', 
                         fontsize=14, fontweight='bold', pad=20)
            ax7.legend(fontsize=11)
            ax7.text(0.02, 0.98, f'n={len(self.reaction_times)} state changes measured', 
                    transform=ax7.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax7.text(0.5, 0.5, 'No reaction time data collected', 
                    transform=ax7.transAxes, fontsize=14, ha='center')
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_7_reaction_times.png', dpi=200, facecolor='white')
        plt.close()
        
        fig8, ax8 = plt.subplots(figsize=(10, 8))
        ax8.axis('off')
        

        
        stats_text = f"""COMPREHENSIVE PERFORMANCE REPORT

══════════════════════════════════════════════════════
SPEED CONTROL METRICS
══════════════════════════════════════════════════════
Mean Speed:              {np.mean(self.speed_history):.3f} m/s
Speed Variance:          {np.var(self.speed_history):.4f}
Mean Tracking Error:     {np.mean(self.speed_error_history):.3f} m/s
Root Mean Square Error:  {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f} m/s
Max Speed Recorded:      {np.max(self.speed_history):.3f} m/s

══════════════════════════════════════════════════════
COLOR DETECTION STATISTICS
══════════════════════════════════════════════════════
Red Light:
  Mean Detection:        {np.mean(self.red_history):.2f}%
  Std Deviation:         {np.std(self.red_history):.2f}%
  Peak Detection:        {np.max(self.red_history):.2f}%

Yellow Light:
  Mean Detection:        {np.mean(self.yellow_history):.2f}%
  Std Deviation:         {np.std(self.yellow_history):.2f}%

Green Light:
  Mean Detection:        {np.mean(self.green_history):.2f}%
  Std Deviation:         {np.std(self.green_history):.2f}%
  Peak Detection:        {np.max(self.green_history):.2f}%

Average Confidence:      {np.mean(self.detection_confidence):.2f}

══════════════════════════════════════════════════════
RESPONSE CHARACTERISTICS
══════════════════════════════════════════════════════
Reaction Time Samples:   {len(self.reaction_times)} measurements
Mean Reaction Time:      {np.mean(self.reaction_times) if self.reaction_times else 0:.2f} seconds
Reaction Time StdDev: {np.std(self.reaction_times) if self.reaction_times else 0:.2f} seconds
SESSION INFORMATION
══════════════════════════════════════════════════════
Total Duration:          {times[-1]:.1f} seconds
Total Frames Processed:  {self.frame_count}
Average Frame Rate:      {self.frame_count/times[-1]:.1f} FPS
"""

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
