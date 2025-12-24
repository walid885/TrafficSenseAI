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

class YOLOTrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('yolo_traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        # Load YOLOv4-tiny
        self.net = cv2.dnn.readNetFromDarknet(
    '/home/raspb/Desktop/TrafficSenseAI/models/yolov4-tiny.cfg',
    '/home/raspb/Desktop/TrafficSenseAI/models/yolov4-tiny.weights'
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Traffic light class IDs (adjust based on your trained model)
        self.TL_RED = 0
        self.TL_YELLOW = 1
        self.TL_GREEN = 2
        
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        
        # History buffers
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
        
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(String, '/traffic_light_state', self.state_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        cv2.namedWindow("YOLO Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("YOLO Traffic Light Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.get_logger().info('YOLO Visualizer - Press Q for analytics')
        
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
    
    def detect_traffic_lights(self, image):
        height, width = image.shape[:2]
        
        # YOLO preprocessing
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detections = {'red': [], 'yellow': [], 'green': []}
        
        if len(indices) > 0:
            for i in indices.flatten():
                if class_ids[i] == self.TL_RED:
                    detections['red'].append((boxes[i], confidences[i]))
                elif class_ids[i] == self.TL_YELLOW:
                    detections['yellow'].append((boxes[i], confidences[i]))
                elif class_ids[i] == self.TL_GREEN:
                    detections['green'].append((boxes[i], confidences[i]))
        
        return detections
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        detections = self.detect_traffic_lights(cv_image)
        
        # Calculate confidence scores
        red_conf = max([c for _, c in detections['red']], default=0.0)
        yellow_conf = max([c for _, c in detections['yellow']], default=0.0)
        green_conf = max([c for _, c in detections['green']], default=0.0)
        
        # Determine state
        max_conf = max(red_conf, yellow_conf, green_conf)
        if max_conf > 0:
            if red_conf == max_conf:
                detected_state = "RED"
            elif yellow_conf == max_conf:
                detected_state = "YELLOW"
            else:
                detected_state = "GREEN"
        else:
            detected_state = self.current_state
        
        # Update state
        if detected_state != self.current_state:
            self.current_state = detected_state
        
        # Set target speed
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        # Calculate confidence ratio
        confs = sorted([red_conf, yellow_conf, green_conf], reverse=True)
        confidence = confs[0] / (confs[1] + 0.01) if len(confs) > 1 else 1.0
        
        speed_error = abs(self.current_speed - self.target_speed)
        elapsed = time.time() - self.start_time
        
        # Store history (convert confidence to percentage scale for compatibility)
        self.red_history.append(red_conf * 100)
        self.green_history.append(green_conf * 100)
        self.yellow_history.append(yellow_conf * 100)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(confidence)
        
        self.plot_red.append(red_conf * 100)
        self.plot_green.append(green_conf * 100)
        self.plot_yellow.append(yellow_conf * 100)
        self.plot_times.append(elapsed)
        
        self.frame_count += 1
        
        # Visualization
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        cam_h, cam_w = 600, 800
        cam_x, cam_y = 50, 50
        cam_resized = cv2.resize(cv_image, (cam_w, cam_h))
        
        # Draw detections on resized frame
        scale_x = cam_w / cv_image.shape[1]
        scale_y = cam_h / cv_image.shape[0]
        
        for color, dets in detections.items():
            box_color = (0, 0, 255) if color == 'red' else (0, 255, 255) if color == 'yellow' else (0, 255, 0)
            for (x, y, w, h), conf in dets:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                cv2.rectangle(cam_resized, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(cam_resized, f"{color.upper()} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        cv2.rectangle(canvas, (cam_x-5, cam_y-5), (cam_x+cam_w+5, cam_y+cam_h+5), (255, 255, 255), 3)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_resized
        cv2.putText(canvas, "YOLO TRAFFIC LIGHT DETECTION", (cam_x, cam_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Plot area
        plot_x, plot_y = 900, 50
        plot_w, plot_h = 950, 350
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (255, 255, 255), 2)
        cv2.putText(canvas, "DETECTION CONFIDENCE TIMELINE (Last 100 frames)", 
                    (plot_x+10, plot_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        max_val = 1.0
        if len(self.plot_red) > 1:
            plot_data = [
                (list(self.plot_red), (0, 0, 255)),
                (list(self.plot_green), (0, 255, 0)),
                (list(self.plot_yellow), (0, 255, 255))
            ]
            
            max_val = max(max(self.plot_red), max(self.plot_green), max(self.plot_yellow), 1.0)
            
            for data, color in plot_data:
                points = []
                for i, val in enumerate(data):
                    x = plot_x + int((i / len(data)) * plot_w)
                    y = plot_y + plot_h - int((val / max_val) * (plot_h - 20))
                    points.append((x, y))
                
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(canvas, points[i], points[i+1], color, 2)
        
        cv2.putText(canvas, "0%", (plot_x-40, plot_y+plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"{max_val:.1f}%", (plot_x-60, plot_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        legend_y = plot_y + plot_h + 20
        cv2.rectangle(canvas, (plot_x, legend_y), (plot_x+20, legend_y+20), (0, 0, 255), -1)
        cv2.putText(canvas, "Red", (plot_x+30, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+120, legend_y), (plot_x+140, legend_y+20), (0, 255, 255), -1)
        cv2.putText(canvas, "Yellow", (plot_x+150, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(canvas, (plot_x+260, legend_y), (plot_x+280, legend_y+20), (0, 255, 0), -1)
        cv2.putText(canvas, "Green", (plot_x+290, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # State display
        state_x = 50
        state_y = cam_y + cam_h + 80
        
        state_color = (0, 255, 0) if self.current_state == "GREEN" else \
                      (0, 255, 255) if self.current_state == "YELLOW" else \
                      (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        
        cv2.rectangle(canvas, (state_x, state_y), (state_x+380, state_y+120), (50, 50, 50), -1)
        cv2.rectangle(canvas, (state_x, state_y), (state_x+380, state_y+120), state_color, 4)
        cv2.putText(canvas, "TRAFFIC STATE", (state_x+90, state_y+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, self.current_state, (state_x+70, state_y+90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, state_color, 4)
        
        
        # Speed control
        speed_x = 900
        speed_y = 450
        
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+950, speed_y+180), (50, 50, 50), -1)
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+950, speed_y+180), (255, 255, 255), 2)
        cv2.putText(canvas, "ROBOT SPEED CONTROL", (speed_x+300, speed_y+35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(canvas, f"Current: {self.current_speed:.3f} m/s", (speed_x+30, speed_y+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, speed_color, 2)
        cv2.putText(canvas, f"Target:  {self.target_speed:.3f} m/s", (speed_x+30, speed_y+120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"Error:   {speed_error:.3f} m/s", (speed_x+30, speed_y+160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
        
        # Speed bar
        bar_x = speed_x + 550
        bar_w = 350
        bar_h = 100
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
        
        # Info bar
        info_y = display_h - 50
        cv2.rectangle(canvas, (0, info_y), (display_w, display_h), (40, 40, 40), -1)
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Detections: R:{len(detections['red'])} Y:{len(detections['yellow'])} G:{len(detections['green'])} | Press Q for Analytics", 
                    (30, info_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("YOLO Traffic Light Detection", canvas)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
    
    def generate_analytics(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data')
            return
        
        times = np.array(list(self.timestamps))
        
        # Speed control plot
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual Speed', linewidth=2.5)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target Speed', linewidth=2.5)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2, color='gray')
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Robot Speed Control Performance', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_1_speed_control.png', dpi=200, facecolor='white')
        plt.close()
        
        # Detection confidence plot
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(times, list(self.red_history), 'r-', label='Red Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.green_history), 'g-', label='Green Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow Light', alpha=0.8, linewidth=2.5)
        
        ax2.set_title('YOLO Traffic Light Detection Confidence', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_2_yolo_detection.png', dpi=200, facecolor='white')
        plt.close()
        
        # Speed error plot
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(times, list(self.speed_error_history), 'orange', linewidth=2)
        ax3.fill_between(times, 0, list(self.speed_error_history), alpha=0.3, color='orange')
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speed Error (m/s)', fontsize=12, fontweight='bold')
        ax3.set_title('Speed Tracking Error', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        mean_error = np.mean(self.speed_error_history)
        ax3.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error:.3f} m/s')
        ax3.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('analytics_3_speed_error.png', dpi=200, facecolor='white')
        plt.close()
        
        # Detection confidence ratio
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(times, list(self.detection_confidence), 'cyan', linewidth=2)
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence Ratio', fontsize=12, fontweight='bold')
        ax4.set_title('YOLO Detection Confidence Ratio', fontsize=14, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_4_detection_confidence.png', dpi=200, facecolor='white')
        plt.close()
        
        # Speed variance
        fig5, ax5 = plt.subplots(figsize=(14, 6))
        window = 50
        if len(self.speed_history) > window:
            variances = [np.std(list(self.speed_history)[max(0, i-window):i+1]) 
                        for i in range(len(self.speed_history))]
            ax5.plot(times, variances, 'purple', linewidth=2)
            ax5.fill_between(times, 0, variances, alpha=0.3, color='purple')
        ax5.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Standard Deviation (m/s)', fontsize=12, fontweight='bold')
        ax5.set_title('Speed Stability', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_5_speed_variance.png', dpi=200, facecolor='white')
        plt.close()
        
        # Speed by state boxplot
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
        ax6.set_title('Speed Distribution by Traffic Light State', fontsize=14, fontweight='bold', pad=20)
        ax6.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('analytics_6_speed_by_state.png', dpi=200, facecolor='white')
        plt.close()
        
        # Reaction times
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        if self.reaction_times:
            ax7.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            mean_rt = np.mean(self.reaction_times)
            ax7.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}s')
            ax7.set_xlabel('Reaction Time (seconds)', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax7.set_title('Robot Reaction Time Distribution', fontsize=14, fontweight='bold', pad=20)
            ax7.legend(fontsize=11)
        else:
            ax7.text(0.5, 0.5, 'No reaction time data', transform=ax7.transAxes, fontsize=14, ha='center')
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_7_reaction_times.png', dpi=200, facecolor='white')
        plt.close()
        
        # Summary stats
        fig8, ax8 = plt.subplots(figsize=(10, 8))
        ax8.axis('off')
        
        stats_text = f"""YOLO TRAFFIC LIGHT DETECTION - PERFORMANCE REPORT

══════════════════════════════════════════════════════
SPEED CONTROL METRICS
══════════════════════════════════════════════════════
Mean Speed:              {np.mean(self.speed_history):.3f} m/s
Speed Variance:          {np.var(self.speed_history):.4f}
Mean Tracking Error:     {np.mean(self.speed_error_history):.3f} m/s
RMSE:                    {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f} m/s
Max Speed:               {np.max(self.speed_history):.3f} m/s

══════════════════════════════════════════════════════
YOLO DETECTION STATISTICS
══════════════════════════════════════════════════════
Red Light:
  Mean Confidence:       {np.mean(self.red_history):.2f}%
  Std Deviation:         {np.std(self.red_history):.2f}%
  Peak Confidence:       {np.max(self.red_history):.2f}%

Yellow Light:
  Mean Confidence:       {np.mean(self.yellow_history):.2f}%
  Std Deviation:         {np.std(self.yellow_history):.2f}%
  Peak Confidence:       {np.max(self.yellow_history):.2f}%

Green Light:
  Mean Confidence:       {np.mean(self.green_history):.2f}%
  Std 
  Deviation:         {np.std(self.green_history):.2f}%
Peak Confidence:       {np.max(self.green_history):.2f}%
Average Confidence Ratio: {np.mean(self.detection_confidence):.2f}
══════════════════════════════════════════════════════
RESPONSE CHARACTERISTICS
══════════════════════════════════════════════════════
Reaction Samples:        {len(self.reaction_times)}
Mean Reaction Time:      {np.mean(self.reaction_times) if self.reaction_times else 0:.2f}s
Reaction StdDev:         {np.std(self.reaction_times) if self.reaction_times else 0:.2f}s
══════════════════════════════════════════════════════
SESSION INFO
══════════════════════════════════════════════════════
Duration:                {times[-1]:.1f}s
Frames Processed:        {self.frame_count}
Average FPS:             {self.frame_count/times[-1]:.1f}
"""
def main():
    rclpy.init()
    node = YOLOTrafficLightVisualizer()
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