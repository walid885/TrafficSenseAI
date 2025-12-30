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

class HybridTrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('hybrid_traffic_light_visualizer')
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
        
        # YOLO class IDs
        self.TL_RED = 0
        self.TL_YELLOW = 1
        self.TL_GREEN = 2
        
        self.conf_threshold = 0.3
        self.nms_threshold = 0.3
        
        # Hybrid weights
        self.yolo_weight = 0.7
        self.hsv_weight = 0.3
        
        # History buffers
        self.red_history = deque(maxlen=2000)
        self.green_history = deque(maxlen=2000)
        self.yellow_history = deque(maxlen=2000)
        self.yolo_red_history = deque(maxlen=2000)
        self.yolo_green_history = deque(maxlen=2000)
        self.yolo_yellow_history = deque(maxlen=2000)
        self.hsv_red_history = deque(maxlen=2000)
        self.hsv_green_history = deque(maxlen=2000)
        self.hsv_yellow_history = deque(maxlen=2000)
        self.hybrid_confidence = deque(maxlen=2000)
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
        
        self.plot_window = 150
        self.plot_red = deque(maxlen=self.plot_window)
        self.plot_green = deque(maxlen=self.plot_window)
        self.plot_yellow = deque(maxlen=self.plot_window)
        self.plot_times = deque(maxlen=self.plot_window)
        
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(String, '/traffic_light_state', self.state_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        cv2.namedWindow("Hybrid Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hybrid Traffic Light Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.get_logger().info('Hybrid YOLO+HSV Visualizer - Press Q for analytics')
        
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
    
    def detect_yolo(self, image):
        height, width = image.shape[:2]
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
        
        red_conf = max([c for _, c in detections['red']], default=0.0)
        yellow_conf = max([c for _, c in detections['yellow']], default=0.0)
        green_conf = max([c for _, c in detections['green']], default=0.0)
        
        return detections, (red_conf, yellow_conf, green_conf)
    
    def detect_hsv(self, image, yolo_detections):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for ROI based on YOLO detections
        roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        has_detections = False
        
        if yolo_detections:
            for color in ['red', 'yellow', 'green']:
                for (x, y, w, h), _ in yolo_detections[color]:
                    has_detections = True
                    # Expand ROI by 30%
                    x_exp = max(0, x - int(w * 0.3))
                    y_exp = max(0, y - int(h * 0.3))
                    w_exp = min(image.shape[1] - x_exp, int(w * 1.6))
                    h_exp = min(image.shape[0] - y_exp, int(h * 1.6))
                    roi_mask[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp] = 255
        
        # If no YOLO detections, use top half of image
        if not has_detections:
            roi_mask[:image.shape[0]//2, :] = 255
        
        # HSV color detection with wider ranges
        red1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        red_mask = (red1 | red2) & roi_mask
        green_mask = cv2.inRange(hsv, (40, 80, 50), (80, 255, 255)) & roi_mask
        yellow_mask = cv2.inRange(hsv, (15, 100, 50), (35, 255, 255)) & roi_mask
        
        total_roi_pixels = cv2.countNonZero(roi_mask)
        if total_roi_pixels > 0:
            red_ratio = (cv2.countNonZero(red_mask) / total_roi_pixels) * 100
            green_ratio = (cv2.countNonZero(green_mask) / total_roi_pixels) * 100
            yellow_ratio = (cv2.countNonZero(yellow_mask) / total_roi_pixels) * 100
        else:
            red_ratio = green_ratio = yellow_ratio = 0.0
        
        return (red_mask, green_mask, yellow_mask), (red_ratio, green_ratio, yellow_ratio)
    
    def hybrid_fusion(self, yolo_scores, hsv_scores):
        # Normalize YOLO scores (0-1) to percentage scale
        yolo_red, yolo_yellow, yolo_green = yolo_scores
        yolo_red_norm = yolo_red * 100
        yolo_yellow_norm = yolo_yellow * 100
        yolo_green_norm = yolo_green * 100
        
        # HSV scores already in percentage
        hsv_red, hsv_yellow, hsv_green = hsv_scores
        
        # Weighted fusion
        hybrid_red = (self.yolo_weight * yolo_red_norm) + (self.hsv_weight * hsv_red)
        hybrid_yellow = (self.yolo_weight * yolo_yellow_norm) + (self.hsv_weight * hsv_yellow)
        hybrid_green = (self.yolo_weight * yolo_green_norm) + (self.hsv_weight * hsv_green)
        
        # Determine state with lower threshold
        max_score = max(hybrid_red, hybrid_yellow, hybrid_green)
        
        if max_score < 2.0:  # Lower threshold
            return "UNKNOWN", (hybrid_red, hybrid_yellow, hybrid_green), 0.0
        
        if hybrid_red == max_score:
            state = "RED"
        elif hybrid_yellow == max_score:
            state = "YELLOW"
        else:
            state = "GREEN"
        
        # Calculate confidence
        sorted_scores = sorted([hybrid_red, hybrid_yellow, hybrid_green], reverse=True)
        confidence = sorted_scores[0] / (sorted_scores[1] + 0.01)
        
        return state, (hybrid_red, hybrid_yellow, hybrid_green), confidence
    
    def publish_speed(self, speed):
        cmd = Twist()
        cmd.linear.x = speed
        self.cmd_pub.publish(cmd)
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run both detectors
        yolo_detections, yolo_scores = self.detect_yolo(cv_image)
        hsv_masks, hsv_scores = self.detect_hsv(cv_image, yolo_detections)
        
        # Hybrid fusion
        detected_state, hybrid_scores, confidence = self.hybrid_fusion(yolo_scores, hsv_scores)
        hybrid_red, hybrid_yellow, hybrid_green = hybrid_scores
        
        # Update state
        if detected_state != "UNKNOWN":
            if self.current_state != detected_state and self.current_state != "UNKNOWN":
                self.last_state_change = time.time()
                self.state_change_speed = self.current_speed
            self.current_state = detected_state
        
        # Set target speed with more variation
        if self.current_state == "GREEN":
            self.target_speed = 0.6
        elif self.current_state == "YELLOW":
            self.target_speed = 0.3
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.4
        
        # Smooth speed control with acceleration
        speed_diff = self.target_speed - self.current_speed
        accel_rate = 0.05  # Acceleration per frame
        
        if abs(speed_diff) > accel_rate:
            if speed_diff > 0:
                self.current_speed += accel_rate
            else:
                self.current_speed -= accel_rate
        else:
            self.current_speed = self.target_speed
        
        self.current_speed = max(0.0, min(0.6, self.current_speed))
        self.publish_speed(self.current_speed)
        
        speed_error = abs(self.current_speed - self.target_speed)
        elapsed = time.time() - self.start_time
        
        # Store history
        self.red_history.append(hybrid_red)
        self.green_history.append(hybrid_green)
        self.yellow_history.append(hybrid_yellow)
        self.yolo_red_history.append(yolo_scores[0] * 100)
        self.yolo_yellow_history.append(yolo_scores[1] * 100)
        self.yolo_green_history.append(yolo_scores[2] * 100)
        self.hsv_red_history.append(hsv_scores[0])
        self.hsv_yellow_history.append(hsv_scores[1])
        self.hsv_green_history.append(hsv_scores[2])
        self.hybrid_confidence.append(confidence)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(confidence)
        
        self.plot_red.append(hybrid_red)
        self.plot_green.append(hybrid_green)
        self.plot_yellow.append(hybrid_yellow)
        self.plot_times.append(elapsed)
        
        self.frame_count += 1
        
        # === VISUALIZATION ===
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Camera feed with YOLO boxes
        cam_h, cam_w = 600, 800
        cam_x, cam_y = 50, 50
        cam_resized = cv2.resize(cv_image, (cam_w, cam_h))
        
        # Draw YOLO detections
        scale_x = cam_w / cv_image.shape[1]
        scale_y = cam_h / cv_image.shape[0]
        
        for color, dets in yolo_detections.items():
            box_color = (0, 0, 255) if color == 'red' else (0, 255, 255) if color == 'yellow' else (0, 255, 0)
            for (x, y, w, h), conf in dets:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                cv2.rectangle(cam_resized, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(cam_resized, f"Y:{conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Add HSV overlay
        red_overlay = cv2.resize(hsv_masks[0], (cam_w, cam_h))
        green_overlay = cv2.resize(hsv_masks[1], (cam_w, cam_h))
        yellow_overlay = cv2.resize(hsv_masks[2], (cam_w, cam_h))
        
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
        cv2.putText(canvas, "HYBRID: YOLO (BOXES) + HSV (OVERLAY)", (cam_x, cam_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timeline plot
        plot_x, plot_y = 900, 50
        plot_w, plot_h = 950, 350
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (255, 255, 255), 2)
        cv2.putText(canvas, f"HYBRID DETECTION TIMELINE (Last {self.plot_window} frames)", 
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
                        cv2.line(canvas, points[i], points[i+1], color, 3)
        
        cv2.putText(canvas, "0%", (plot_x-40, plot_y+plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"{max_val:.0f}%", (plot_x-60, plot_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
        
        # Metrics comparison
        metrics_x = 470
        metrics_y = state_y
        
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+380, metrics_y+240), (50, 50, 50), -1)
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+380, metrics_y+240), (255, 255, 255), 2)
        
        cv2.putText(canvas, "HYBRID SCORES", (metrics_x+100, metrics_y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        y_offset = metrics_y + 60
        cv2.putText(canvas, f"R: {hybrid_red:5.2f}%", (metrics_x+20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(canvas, f"Y={yolo_scores[0]*100:.1f} H={hsv_scores[0]:.1f}", 
                    (metrics_x+190, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        y_offset += 40
        cv2.putText(canvas, f"Y: {hybrid_yellow:5.2f}%", (metrics_x+20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(canvas, f"Y={yolo_scores[1]*100:.1f} H={hsv_scores[1]:.1f}", 
                    (metrics_x+190, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        y_offset += 40
        cv2.putText(canvas, f"G: {hybrid_green:5.2f}%", (metrics_x+20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas, f"Y={yolo_scores[2]*100:.1f} H={hsv_scores[2]:.1f}", 
                    (metrics_x+190, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        y_offset += 50
        cv2.putText(canvas, f"Confidence: {confidence:.2f}", (metrics_x+70, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        cv2.putText(canvas, f"Weights: Y={self.yolo_weight} H={self.hsv_weight}", 
                    (metrics_x+50, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
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
        yolo_count = len(yolo_detections['red']) + len(yolo_detections['yellow']) + len(yolo_detections['green'])
        cv2.putText(canvas, f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | YOLO: {yolo_count} | Conf: {confidence:.2f} | Threshold: 2.0 | Press Q for Analytics", 
                    (30, info_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Hybrid Traffic Light Detection", canvas)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_analytics()
            raise KeyboardInterrupt
    
    def generate_analytics(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data')
            return
        
        times = np.array(list(self.timestamps))
        
        # Speed control
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(times, list(self.speed_history), 'b-', label='Actual Speed', linewidth=2.5)
        ax1.plot(times, list(self.target_speed_history), 'r--', label='Target Speed', linewidth=2.5)
        ax1.fill_between(times, list(self.speed_history), list(self.target_speed_history), alpha=0.2, color='gray')
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Robot Speed Control Performance - Hybrid System', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hybrid_analytics_1_speed_control.png', dpi=200, facecolor='white')
        plt.close()
        
        # Hybrid detection comparison
        fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(14, 12))
        
        ax2a.plot(times, list(self.red_history), 'r-', label='Hybrid Red', linewidth=2.5)
        ax2a.plot(times, list(self.yolo_red_history), 'r--', alpha=0.6, label='YOLO Red', linewidth=1.5)
        ax2a.plot(times, list(self.hsv_red_history), 'r:', alpha=0.6, label='HSV Red', linewidth=1.5)
        ax2a.set_ylabel('Red Detection (%)', fontsize=11, fontweight='bold')
        ax2a.set_title('Hybrid vs YOLO vs HSV - Red Light Detection', fontsize=13, fontweight='bold')
        ax2a.legend(fontsize=10)
        ax2a.grid(True, alpha=0.3)
        
        ax2b.plot(times, list(self.yellow_history), 'y-', label='Hybrid Yellow', linewidth=2.5)
        ax2b.plot(times, list(self.yolo_yellow_history), 'y--', alpha=0.6, label='YOLO Yellow', linewidth=1.5)
        ax2b.plot(times, list(self.hsv_yellow_history), 'y:', alpha=0.6, label='HSV Yellow', linewidth=1.5)
        ax2b.set_ylabel('Yellow Detection (%)', fontsize=11, fontweight='bold')
        ax2b.set_title('Hybrid vs YOLO vs HSV - Yellow Light Detection', fontsize=13, fontweight='bold')
        ax2b.legend(fontsize=10)
        ax2b.grid(True, alpha=0.3)
        
        ax2c.plot(times, list(self.green_history), 'g-', label='Hybrid Green', linewidth=2.5)
        ax2c.plot(times, list(self.yolo_green_history), 'g--', alpha=0.6, label='YOLO Green', linewidth=1.5)
        ax2c.plot(times, list(self.hsv_green_history), 'g:', alpha=0.6, label='HSV Green', linewidth=1.5)
        ax2c.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax2c.set_ylabel('Green Detection (%)', fontsize=11, fontweight='bold')
        ax2c.set_title('Hybrid vs YOLO vs HSV - Green Light Detection', fontsize=13, fontweight='bold')
        ax2c.legend(fontsize=10)
        ax2c.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hybrid_analytics_2_detection_comparison.png', dpi=200, facecolor='white')
        plt.close()
        
        # Speed error
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(times, list(self.speed_error_history), 'orange', linewidth=2)
        ax3.fill_between(times, 0, list(self.speed_error_history), alpha=0.3, color='orange')
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speed Error (m/s)', fontsize=12, fontweight='bold')
        ax3.set_title('Speed Tracking Error - Hybrid System', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        mean_error = np.mean(self.speed_error_history)
        ax3.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error:.3f} m/s')
        ax3.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('hybrid_analytics_3_speed_error.png', dpi=200, facecolor='white')
        plt.close()
        
        # Hybrid confidence
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(times, list(self.hybrid_confidence), 'cyan', linewidth=2)
        ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Confidence Threshold')
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Confidence Ratio', fontsize=12, fontweight='bold')
        ax4.set_title('Hybrid Detection Confidence Over Time', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hybrid_analytics_4_confidence.png', dpi=200, facecolor='white')
        plt.close()
        
        # Method correlation analysis
        fig5, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        axes[0].scatter(self.yolo_red_history, self.hsv_red_history, alpha=0.5, c='red', s=10)
        axes[0].set_xlabel('YOLO Red Score', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('HSV Red Score', fontsize=10, fontweight='bold')
        axes[0].set_title('Red Light: YOLO vs HSV Correlation', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(self.yolo_yellow_history, self.hsv_yellow_history, alpha=0.5, c='yellow', s=10)
        axes[1].set_xlabel('YOLO Yellow Score', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('HSV Yellow Score', fontsize=10, fontweight='bold')
        axes[1].set_title('Yellow Light: YOLO vs HSV Correlation', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].scatter(self.yolo_green_history, self.hsv_green_history, alpha=0.5, c='green', s=10)
        axes[2].set_xlabel('YOLO Green Score', fontsize=10, fontweight='bold')
        axes[2].set_ylabel('HSV Green Score', fontsize=10, fontweight='bold')
        axes[2].set_title('Green Light: YOLO vs HSV Correlation', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hybrid_analytics_5_correlation.png', dpi=200, facecolor='white')
        plt.close()
        
        # Speed by state
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
        ax6.set_title('Speed Distribution by Traffic Light State - Hybrid', fontsize=14, fontweight='bold', pad=20)
        ax6.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('hybrid_analytics_6_speed_by_state.png', dpi=200, facecolor='white')
        plt.close()
        
        # Reaction times
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        if self.reaction_times:
            ax7.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            mean_rt = np.mean(self.reaction_times)
            ax7.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}s')
            ax7.set_xlabel('Reaction Time (seconds)', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax7.set_title('Robot Reaction Time Distribution - Hybrid System', fontsize=14, fontweight='bold', pad=20)
            ax7.legend(fontsize=11)
        else:
            ax7.text(0.5, 0.5, 'No reaction time data', transform=ax7.transAxes, fontsize=14, ha='center')
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hybrid_analytics_7_reaction_times.png', dpi=200, facecolor='white')
        plt.close()
        
        # Summary report
        fig8, ax8 = plt.subplots(figsize=(12, 10))
        ax8.axis('off')
        
        # Calculate correlation coefficients
        yolo_red_arr = np.array(list(self.yolo_red_history))
        hsv_red_arr = np.array(list(self.hsv_red_history))
        yolo_yellow_arr = np.array(list(self.yolo_yellow_history))
        hsv_yellow_arr = np.array(list(self.hsv_yellow_history))
        yolo_green_arr = np.array(list(self.yolo_green_history))
        hsv_green_arr = np.array(list(self.hsv_green_history))
        
        corr_red = np.corrcoef(yolo_red_arr, hsv_red_arr)[0,1] if len(yolo_red_arr) > 1 else 0
        corr_yellow = np.corrcoef(yolo_yellow_arr, hsv_yellow_arr)[0,1] if len(yolo_yellow_arr) > 1 else 0
        corr_green = np.corrcoef(yolo_green_arr, hsv_green_arr)[0,1] if len(yolo_green_arr) > 1 else 0
        
        stats_text = f"""HYBRID TRAFFIC LIGHT DETECTION - COMPREHENSIVE REPORT
Fusion: YOLO (70%) + HSV (30%)

══════════════════════════════════════════════════════
SPEED CONTROL METRICS
══════════════════════════════════════════════════════
Mean Speed:              {np.mean(self.speed_history):.3f} m/s
Speed Variance:          {np.var(self.speed_history):.4f}
Mean Tracking Error:     {np.mean(self.speed_error_history):.3f} m/s
RMSE:                    {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f} m/s
Max Speed:               {np.max(self.speed_history):.3f} m/s

══════════════════════════════════════════════════════
HYBRID DETECTION STATISTICS
══════════════════════════════════════════════════════
Red Light (Hybrid):
  Mean Score:            {np.mean(self.red_history):.2f}%
  Std Deviation:         {np.std(self.red_history):.2f}%
  Peak Score:            {np.max(self.red_history):.2f}%
  YOLO Contribution:     {np.mean(self.yolo_red_history):.2f}%
  HSV Contribution:      {np.mean(self.hsv_red_history):.2f}%
  Correlation (Y-H):     {corr_red:.3f}

Yellow Light (Hybrid):
  Mean Score:            {np.mean(self.yellow_history):.2f}%
  Std Deviation:         {np.std(self.yellow_history):.2f}%
  Peak Score:            {np.max(self.yellow_history):.2f}%
  YOLO Contribution:     {np.mean(self.yolo_yellow_history):.2f}%
  HSV Contribution:      {np.mean(self.hsv_yellow_history):.2f}%
  Correlation (Y-H):     {corr_yellow:.3f}

Green Light (Hybrid):
  Mean Score:            {np.mean(self.green_history):.2f}%
  Std Deviation:         {np.std(self.green_history):.2f}%
  Peak Score:            {np.max(self.green_history):.2f}%
  YOLO Contribution:     {np.mean(self.yolo_green_history):.2f}%
  HSV Contribution:      {np.mean(self.hsv_green_history):.2f}%
  Correlation (Y-H):     {corr_green:.3f}

Average Confidence:      {np.mean(self.hybrid_confidence):.2f}

══════════════════════════════════════════════════════
RESPONSE CHARACTERISTICS
══════════════════════════════════════════════════════
Reaction Samples:        {len(self.reaction_times)}
Mean Reaction Time:      {np.mean(self.reaction_times) if self.reaction_times else 0:.2f}s
Reaction StdDev:         {np.std(self.reaction_times) if self.reaction_times else 0:.2f}s

══════════════════════════════════════════════════════
SESSION INFORMATION
══════════════════════════════════════════════════════
Duration:                {times[-1]:.1f}s
Frames Processed:        {self.frame_count}
Average FPS:             {self.frame_count/times[-1]:.1f}

══════════════════════════════════════════════════════
FUSION PARAMETERS
══════════════════════════════════════════════════════
YOLO Weight:             {self.yolo_weight}
HSV Weight:              {self.hsv_weight}
Detection Threshold:     2.0%
Confidence Threshold:    2.0
YOLO Conf Threshold:     {self.conf_threshold}
NMS Threshold:           {self.nms_threshold}
"""
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('hybrid_analytics_8_summary.png', dpi=200, facecolor='white')
        plt.close()
        
        self.get_logger().info('Analytics saved: 8 PNG files generated')

def main():
    rclpy.init()
    node = HybridTrafficLightVisualizer()
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