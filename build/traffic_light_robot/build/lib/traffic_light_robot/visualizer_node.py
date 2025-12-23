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
        
        # STATE PERSISTENCE - aggressive state changes
        self.state_buffer = deque(maxlen=5)
        self.min_confidence = 0.0001  # Accept ANY detection above noise
        
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
            Image, '/camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        
        cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Traffic Light Detection", 1920, 1080)
        
        self.get_logger().info('Visualizer with Aggressive Detection - Press Q for analytics')
        
    def state_callback(self, msg):
        if self.current_state != msg.data and self.current_state != "UNKNOWN":
            elapsed = time.time() - self.start_time
            self.last_state_change = time.time()
            self.state_change_speed = self.current_speed
            
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
        
        # RELAXED THRESHOLDS - detect any color presence
        red1 = cv2.inRange(hsv, (0, 30, 50), (20, 255, 255))
        red2 = cv2.inRange(hsv, (160, 30, 50), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (35, 30, 50), (85, 255, 255))
        yellow_mask = cv2.inRange(hsv, (15, 30, 50), (35, 255, 255))
        
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        
        ratios = {'RED': red_ratio, 'YELLOW': yellow_ratio, 'GREEN': green_ratio}
        max_ratio = max(ratios.values())
        
        # DETERMINE STATE - accept even tiny detections
        if max_ratio > self.min_confidence:
            detected = max(ratios, key=ratios.get)
        else:
            detected = "UNKNOWN"
        
        # BUFFERED VOTING - 3/5 frames agree
        self.state_buffer.append(detected)
        if len(self.state_buffer) >= 3:
            counts = {s: list(self.state_buffer).count(s) for s in set(self.state_buffer)}
            voted_state = max(counts, key=counts.get)
            
            if counts[voted_state] >= 3:
                self.current_state = voted_state
        
        # SET TARGET SPEED
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
                    "dominant_color": detected,
                    "confidence": round(max_ratio, 6)
                },
                "speed_control": {
                    "current_speed": round(self.current_speed, 4),
                    "target_speed": round(self.target_speed, 4),
                    "speed_error": round(speed_error, 4)
                },
                "traffic_state": self.current_state,
                "state_buffer": list(self.state_buffer)
            }
            self.json_log["frame_data"].append(frame_data)
        
        self.red_history.append(red_ratio)
        self.green_history.append(green_ratio)
        self.yellow_history.append(yellow_ratio)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(max_ratio)
        
        self.plot_red.append(red_ratio)
        self.plot_green.append(green_ratio)
        self.plot_yellow.append(yellow_ratio)
        self.plot_times.append(elapsed)
        
        self.frame_count += 1
        
        # VISUALIZATION
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
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
        
        plot_x, plot_y = 1070, 30
        plot_w, plot_h = 800, 400
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (255, 255, 255), 2)
        
        cv2.putText(canvas, "CONFIDENCE TIMELINE", 
                    (plot_x+10, plot_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        max_val = 0.02
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
        
        cv2.putText(canvas, "DETECTION RATIOS", (metrics_x+120, metrics_y+40), 
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
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Buffer: {list(self.state_buffer)} | Press Q for Analytics", 
                    (30, info_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Traffic Light Detection", canvas)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
        
    def save_json_log(self, filename='traffic_light_detection_log.json'):
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
        
        self.save_json_log('traffic_light_detection_log.json')
        
        times = np.array(list(self.timestamps))
        
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual Speed', linewidth=2.5)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target Speed', linewidth=2.5)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2, color='gray')
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Robot Speed Control Performance', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_1_speed_control.png', dpi=200)
        plt.close()
        
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(times, list(self.red_history), 'r-', label='Red', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.green_history), 'g-', label='Green', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow', alpha=0.8, linewidth=2.5)
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Detection Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Color Detection Over Time', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_2_color_detection.png', dpi=200)
        plt.close()
        
        print("\n" + "="*80)
        print("ANALYTICS SAVED")
        print("="*80)

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