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
        self.hsv_state = "UNKNOWN"
        self.yolo_state = "UNKNOWN"
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
        
        self.TL_RED = 0
        self.TL_YELLOW = 1
        self.TL_GREEN = 2
        
        self.conf_threshold = 0.3
        self.nms_threshold = 0.3
        self.yolo_variation_factor = 0.05

        # History buffers
        self.red_history = deque(maxlen=2000)
        self.green_history = deque(maxlen=2000)
        self.yellow_history = deque(maxlen=2000)
        self.yolo_red_history = deque(maxlen=2000)
        self.yolo_green_history = deque(maxlen=2000)
        self.yolo_yellow_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.target_speed_history = deque(maxlen=2000)
        self.timestamps = deque(maxlen=2000)
        self.state_history = deque(maxlen=2000)
        self.hsv_state_history = deque(maxlen=2000)
        self.yolo_state_history = deque(maxlen=2000)
        self.speed_error_history = deque(maxlen=2000)
        self.detection_confidence = deque(maxlen=2000)
        self.reaction_times = []
        self.yolo_state_buffer = deque(maxlen=10)
        self.yolo_needs_hsv = False
        self.yolo_hsv_capture_time = None
        self.yolo_delay_duration = 0.5

        self.frame_count = 0
        self.start_time = time.time()
        self.last_state_change = None
        self.state_change_speed = None
        
        self.plot_window = 100
        self.plot_red = deque(maxlen=self.plot_window)
        self.plot_green = deque(maxlen=self.plot_window)
        self.plot_yellow = deque(maxlen=self.plot_window)
        self.plot_yolo_red = deque(maxlen=self.plot_window)
        self.plot_yolo_green = deque(maxlen=self.plot_window)
        self.plot_yolo_yellow = deque(maxlen=self.plot_window)
        self.plot_times = deque(maxlen=self.plot_window)
        
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        cv2.namedWindow("Hybrid Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hybrid Traffic Light Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.get_logger().info('Hybrid Visualizer - Press Q for analytics')
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
        if self.last_state_change and self.state_change_speed is not None:
            if abs(self.current_speed - self.target_speed) < 0.05:
                reaction_time = time.time() - self.last_state_change
                if reaction_time < 5.0:
                    self.reaction_times.append(reaction_time)
                self.last_state_change = None
    
    def detect_traffic_lights_yolo(self, image):
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
                varied_confidence = confidences[i] * (1.0 + np.random.uniform(-self.yolo_variation_factor, self.yolo_variation_factor))
                varied_confidence = np.clip(varied_confidence, 0.0, 1.0)
                
                if class_ids[i] == self.TL_RED:
                    detections['red'].append((boxes[i], varied_confidence))
                elif class_ids[i] == self.TL_YELLOW:
                    detections['yellow'].append((boxes[i], varied_confidence))
                elif class_ids[i] == self.TL_GREEN:
                    detections['green'].append((boxes[i], varied_confidence))
        
        return detections
        
    def detect_traffic_lights_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (45, 100, 70), (75, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 120, 70), (30, 255, 255))
        
        total_pixels = image.shape[0] * image.shape[1]
        red_ratio = (cv2.countNonZero(red_mask) / total_pixels) * 100
        green_ratio = (cv2.countNonZero(green_mask) / total_pixels) * 100
        yellow_ratio = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
        
        return red_ratio, green_ratio, yellow_ratio, red_mask, green_mask, yellow_mask
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # YOLO detection
        yolo_detections = self.detect_traffic_lights_yolo(cv_image)
        yolo_red_conf = max([c for _, c in yolo_detections['red']], default=0.0)
        yolo_yellow_conf = max([c for _, c in yolo_detections['yellow']], default=0.0)
        yolo_green_conf = max([c for _, c in yolo_detections['green']], default=0.0)
        
        yolo_max_conf = max(yolo_red_conf, yolo_yellow_conf, yolo_green_conf)
        
        # HSV detection
        hsv_red_ratio, hsv_green_ratio, hsv_yellow_ratio, red_mask, green_mask, yellow_mask = self.detect_traffic_lights_hsv(cv_image)
        
        hsv_max_ratio = max(hsv_red_ratio, hsv_green_ratio, hsv_yellow_ratio)
        if hsv_red_ratio == hsv_max_ratio:
            self.hsv_state = "RED"
        elif hsv_yellow_ratio == hsv_max_ratio:
            self.hsv_state = "YELLOW"
        else:
            self.hsv_state = "GREEN"
        
        current_time = time.time()
        self.yolo_state_buffer.append((current_time, self.hsv_state))
        
        # YOLO state update logic
        if yolo_max_conf > 0:
            if yolo_red_conf == yolo_max_conf:
                self.yolo_state = "RED"
            elif yolo_yellow_conf == yolo_max_conf:
                self.yolo_state = "YELLOW"
            else:
                self.yolo_state = "GREEN"
            
            self.yolo_needs_hsv = False
            self.yolo_hsv_capture_time = None
        else:
            if not self.yolo_needs_hsv:
                self.yolo_needs_hsv = True
                self.yolo_hsv_capture_time = current_time
            else:
                time_elapsed = current_time - self.yolo_hsv_capture_time
                
                if time_elapsed >= self.yolo_delay_duration:
                    if len(self.yolo_state_buffer) >= 5:
                        recent_states = [state for _, state in list(self.yolo_state_buffer)[-5:]]
                        
                        if len(set(recent_states)) == 1:
                            self.yolo_state = recent_states[0]
                            self.get_logger().info(f'YOLO adopted stable HSV state: {self.yolo_state}')
                        else:
                            most_common = max(set(recent_states), key=recent_states.count)
                            self.yolo_state = most_common
                            self.get_logger().info(f'YOLO adopted most common HSV state: {self.yolo_state}')
                
                self.yolo_needs_hsv = False
                self.yolo_hsv_capture_time = None

            self.yolo_state = self.hsv_state

        # Hybrid fusion
        prev_state = self.current_state

        if self.yolo_state == self.hsv_state:
            new_state = self.yolo_state
        elif self.yolo_state != "UNKNOWN":
            if yolo_max_conf > 0.5:
                new_state = self.yolo_state
            else:
                new_state = self.hsv_state
        else:
            new_state = self.hsv_state

        if new_state != prev_state:
            self.last_state_change = time.time()
            self.state_change_speed = self.current_speed
            self.current_state = new_state
        
        # Set target speed
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        # Calculate confidence
        hsv_ratios = sorted([hsv_red_ratio, hsv_green_ratio, hsv_yellow_ratio], reverse=True)
        hsv_confidence = hsv_ratios[0] / (hsv_ratios[1] + 0.01)
        
        yolo_confs = sorted([yolo_red_conf, yolo_yellow_conf, yolo_green_conf], reverse=True)
        yolo_confidence = yolo_confs[0] / (yolo_confs[1] + 0.01) if yolo_confs[1] > 0 else 1.0
        
        combined_confidence = (hsv_confidence + yolo_confidence) / 2.0
        
        speed_error = abs(self.current_speed - self.target_speed)
        elapsed = time.time() - self.start_time
        
        # Store history
        self.red_history.append(hsv_red_ratio)
        self.green_history.append(hsv_green_ratio)
        self.yellow_history.append(hsv_yellow_ratio)
        self.yolo_red_history.append(yolo_red_conf * 100)
        self.yolo_green_history.append(yolo_green_conf * 100)
        self.yolo_yellow_history.append(yolo_yellow_conf * 100)
        self.speed_history.append(self.current_speed)
        self.target_speed_history.append(self.target_speed)
        self.timestamps.append(elapsed)
        self.state_history.append(self.current_state)
        self.hsv_state_history.append(self.hsv_state)
        self.yolo_state_history.append(self.yolo_state)
        self.speed_error_history.append(speed_error)
        self.detection_confidence.append(combined_confidence)
        
        self.plot_red.append(hsv_red_ratio)
        self.plot_green.append(hsv_green_ratio)
        self.plot_yellow.append(hsv_yellow_ratio)
        self.plot_yolo_red.append(yolo_red_conf * 100)
        self.plot_yolo_green.append(yolo_green_conf * 100)
        self.plot_yolo_yellow.append(yolo_yellow_conf * 100)
        self.plot_times.append(elapsed)
        
        self.frame_count += 1
        
        # Visualization
        display_h, display_w = 1080, 1920
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 25)  # Dark navy background
        
        cam_h, cam_w = 450, 600
        cam_x, cam_y = 50, 50
        cam_resized = cv2.resize(cv_image, (cam_w, cam_h))
        
        # Draw YOLO detections with enhanced styling
        scale_x = cam_w / cv_image.shape[1]
        scale_y = cam_h / cv_image.shape[0]
        
        for color, dets in yolo_detections.items():
            if color == 'red':
                box_color = (50, 50, 255)
            elif color == 'yellow':
                box_color = (50, 255, 255)
            else:
                box_color = (50, 255, 100)
                
            for (x, y, w, h), conf in dets:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                cv2.rectangle(cam_resized, (x1, y1), (x2, y2), box_color, 3)
                
                # Enhanced label background
                label = f"{conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(cam_resized, (x1, y1-label_size[1]-8), (x1+label_size[0]+8, y1), box_color, -1)
                cv2.putText(cam_resized, label, (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Gradient border for camera
        cv2.rectangle(canvas, (cam_x-6, cam_y-6), (cam_x+cam_w+6, cam_y+cam_h+6), (80, 80, 120), 6)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_resized
        cv2.putText(canvas, "CAMERA + YOLO", (cam_x+5, cam_y-18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 255), 2)
        
        # HSV masks with enhanced borders
        mask_h, mask_w = 450, 600
        mask_x, mask_y = 700, 50
        
        red_overlay_resized = cv2.resize(red_mask, (mask_w, mask_h))
        green_overlay_resized = cv2.resize(green_mask, (mask_w, mask_h))
        yellow_overlay_resized = cv2.resize(yellow_mask, (mask_w, mask_h))
        
        mask_bgr = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        mask_bgr[:,:,2] = red_overlay_resized
        mask_bgr[:,:,1] = cv2.addWeighted(green_overlay_resized, 1.0, yellow_overlay_resized, 1.0, 0)
        
        cv2.rectangle(canvas, (mask_x-6, mask_y-6), (mask_x+mask_w+6, mask_y+mask_h+6), (80, 120, 80), 6)
        canvas[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = mask_bgr
        cv2.putText(canvas, "HSV COLOR MASKS", (mask_x+5, mask_y-18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 255, 200), 2)
        
        # HSV plot with enhanced styling
        plot1_x, plot1_y = 1350, 50
        plot1_w, plot1_h = 520, 200
        
        cv2.rectangle(canvas, (plot1_x, plot1_y), (plot1_x+plot1_w, plot1_y+plot1_h), (25, 30, 35), -1)
        cv2.rectangle(canvas, (plot1_x, plot1_y), (plot1_x+plot1_w, plot1_y+plot1_h), (100, 120, 140), 3)
        cv2.putText(canvas, "HSV DETECTION", (plot1_x+15, plot1_y-12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 200, 255), 2)
        
        # Grid lines for plot
        for i in range(1, 5):
            y_pos = plot1_y + int(i * plot1_h / 5)
            cv2.line(canvas, (plot1_x, y_pos), (plot1_x+plot1_w, y_pos), (40, 50, 60), 1)
        
        if len(self.plot_red) > 1:
            max_val = max(max(self.plot_red), max(self.plot_green), max(self.plot_yellow), 1.0)
            
            for data, color, thickness in [(list(self.plot_red), (80, 80, 255), 3), 
                                            (list(self.plot_green), (80, 255, 120), 3), 
                                            (list(self.plot_yellow), (80, 255, 255), 3)]:
                points = [(plot1_x + int((i / len(data)) * plot1_w), 
                          plot1_y + plot1_h - int((val / max_val) * (plot1_h - 20))) 
                         for i, val in enumerate(data)]
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(canvas, points[i], points[i+1], color, thickness)
        
        # YOLO plot with enhanced styling
        plot2_x, plot2_y = 1350, 300
        plot2_w, plot2_h = 520, 200
        
        cv2.rectangle(canvas, (plot2_x, plot2_y), (plot2_x+plot2_w, plot2_y+plot2_h), (25, 30, 35), -1)
        cv2.rectangle(canvas, (plot2_x, plot2_y), (plot2_x+plot2_w, plot2_y+plot2_h), (140, 100, 120), 3)
        cv2.putText(canvas, "YOLO DETECTION", (plot2_x+15, plot2_y-12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 150), 2)
        
        # Grid lines for YOLO plot
        for i in range(1, 5):
            y_pos = plot2_y + int(i * plot2_h / 5)
            cv2.line(canvas, (plot2_x, y_pos), (plot2_x+plot2_w, y_pos), (40, 50, 60), 1)
        
        if len(self.plot_yolo_red) > 1:
            max_val = max(max(self.plot_yolo_red), max(self.plot_yolo_green), max(self.plot_yolo_yellow), 1.0)
            
            for data, color, thickness in [(list(self.plot_yolo_red), (100, 100, 255), 3), 
                                            (list(self.plot_yolo_green), (100, 255, 140), 3), 
                                            (list(self.plot_yolo_yellow), (100, 255, 255), 3)]:
                points = [(plot2_x + int((i / len(data)) * plot2_w), 
                          plot2_y + plot2_h - int((val / max_val) * (plot2_h - 20))) 
                         for i, val in enumerate(data)]
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(canvas, points[i], points[i+1], color, thickness)
        
        # State displays with glowing effect
        state_y = 550
        state_w = 280
        state_h = 100
        
        # HSV State
        hsv_state_x = 50
        hsv_state_color = (80, 255, 120) if self.hsv_state == "GREEN" else (80, 255, 255) if self.hsv_state == "YELLOW" else (80, 80, 255) if self.hsv_state == "RED" else (100, 100, 100)
        
        cv2.rectangle(canvas, (hsv_state_x, state_y), (hsv_state_x+state_w, state_y+state_h), (30, 35, 40), -1)
        cv2.rectangle(canvas, (hsv_state_x, state_y), (hsv_state_x+state_w, state_y+state_h), hsv_state_color, 5)
        cv2.putText(canvas, "HSV STATE", (hsv_state_x+65, state_y+32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, self.hsv_state, (hsv_state_x+50, state_y+75), cv2.FONT_HERSHEY_DUPLEX, 1.3, hsv_state_color, 3)
        
        # YOLO State
        yolo_state_x = 380
        yolo_state_color = (80, 255, 120) if self.yolo_state == "GREEN" else (80, 255, 255) if self.yolo_state == "YELLOW" else (80, 80, 255) if self.yolo_state == "RED" else (100, 100, 100)
        
        cv2.rectangle(canvas, (yolo_state_x, state_y), (yolo_state_x+state_w, state_y+state_h), (30, 35, 40), -1)
        cv2.rectangle(canvas, (yolo_state_x, state_y), (yolo_state_x+state_w, state_y+state_h), yolo_state_color, 5)
        cv2.putText(canvas, "YOLO STATE", (yolo_state_x+55, state_y+32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, self.yolo_state, (yolo_state_x+45, state_y+75), cv2.FONT_HERSHEY_DUPLEX, 1.3, yolo_state_color, 3)
        
        # Final Hybrid State with stronger emphasis
        final_state_x = 710
        final_state_color = (100, 255, 140) if self.current_state == "GREEN" else (100, 255, 255) if self.current_state == "YELLOW" else (100, 100, 255) if self.current_state == "RED" else (100, 100, 100)
        
        cv2.rectangle(canvas, (final_state_x, state_y), (final_state_x+state_w, state_y+state_h), (35, 40, 45), -1)
        cv2.rectangle(canvas, (final_state_x, state_y), (final_state_x+state_w, state_y+state_h), final_state_color, 7)
        cv2.putText(canvas, "HYBRID STATE", (final_state_x+45, state_y+32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(canvas, self.current_state, (final_state_x+35, state_y+75), cv2.FONT_HERSHEY_DUPLEX, 1.3, final_state_color, 3)
        
        # Metrics with enhanced background
        metrics_x = 1040
        metrics_y = state_y
        
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+380, metrics_y+state_h), (30, 35, 40), -1)
        cv2.rectangle(canvas, (metrics_x, metrics_y), (metrics_x+380, metrics_y+state_h), (120, 140, 160), 3)
        cv2.putText(canvas, "HSV% | YOLO", (metrics_x+105, metrics_y+24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, f"R: {hsv_red_ratio:5.2f} | {yolo_red_conf:.2f}", (metrics_x+20, metrics_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 255), 2)
        cv2.putText(canvas, f"Y: {hsv_yellow_ratio:5.2f} | {yolo_yellow_conf:.2f}", (metrics_x+20, metrics_y+72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 255), 2)
        cv2.putText(canvas, f"G: {hsv_green_ratio:5.2f} | {yolo_green_conf:.2f}", (metrics_x+20, metrics_y+94), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 140), 2)

        # Speed control
        speed_x = 50
        speed_y = 700
        
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+1820, speed_y+180), (50, 50, 50), -1)
        cv2.rectangle(canvas, (speed_x, speed_y), (speed_x+1820, speed_y+180), (255, 255, 255), 2)
        cv2.putText(canvas, "ROBOT SPEED CONTROL", (speed_x+650, speed_y+35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        speed_color = (0, 255, 0) if self.current_speed > 0.1 else (128, 128, 128)
        cv2.putText(canvas, f"Current: {self.current_speed:.3f} m/s", (speed_x+30, speed_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, speed_color, 2)
        cv2.putText(canvas, f"Target:  {self.target_speed:.3f} m/s", (speed_x+30, speed_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"Error:   {speed_error:.3f} m/s", (speed_x+30, speed_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
        
        # Speed bar
        bar_x = speed_x + 550
        bar_w = 1200
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
        agreement = "AGREE" if self.hsv_state == self.yolo_state else "CONFLICT"
        agreement_color = (0, 255, 0) if agreement == "AGREE" else (0, 165, 255)
        info_text = f"Time: {elapsed:.1f}s | Frames: {self.frame_count} | Systems: {agreement} | Confidence: {combined_confidence:.2f} | Press Q for Analytics"
        cv2.putText(canvas, info_text, (30, info_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        ax1.set_title('Robot Speed Control Performance', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_1_speed_control.png', dpi=200, facecolor='white')
        plt.close()
        
        # HSV Detection
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(times, list(self.red_history), 'r-', label='Red Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.green_history), 'g-', label='Green Light', alpha=0.8, linewidth=2.5)
        ax2.plot(times, list(self.yellow_history), 'y-', label='Yellow Light', alpha=0.8, linewidth=2.5)
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('HSV Color Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_title('HSV Traffic Light Detection', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_2_hsv_detection.png', dpi=200, facecolor='white')
        plt.close()
    # YOLO Detection
    # YOLO Detection
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(times, list(self.yolo_red_history), 'r-', label='Red Light', alpha=0.8, linewidth=2.5)
        ax3.plot(times, list(self.yolo_green_history), 'g-', label='Green Light', alpha=0.8, linewidth=2.5)
        ax3.plot(times, list(self.yolo_yellow_history), 'y-', label='Yellow Light', alpha=0.8, linewidth=2.5)
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('YOLO Confidence (%)', fontsize=12, fontweight='bold')
        ax3.set_title('YOLO Traffic Light Detection', fontsize=14, fontweight='bold', pad=20)
        ax3.legend(fontsize=11, loc='upper right')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_3_yolo_detection.png', dpi=200, facecolor='white')
        plt.close()

        # Speed error
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(times, list(self.speed_error_history), 'orange', linewidth=2)
        ax4.fill_between(times, 0, list(self.speed_error_history), alpha=0.3, color='orange')
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Speed Error (m/s)', fontsize=12, fontweight='bold')
        ax4.set_title('Speed Tracking Error', fontsize=14, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3)
        mean_error = np.mean(self.speed_error_history)
        ax4.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error:.3f} m/s')
        ax4.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('analytics_4_speed_error.png', dpi=200, facecolor='white')
        plt.close()

        # Combined confidence
        fig5, ax5 = plt.subplots(figsize=(14, 6))
        ax5.plot(times, list(self.detection_confidence), 'cyan', linewidth=2)
        ax5.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Confidence Ratio', fontsize=12, fontweight='bold')
        ax5.set_title('Combined Detection Confidence', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_5_detection_confidence.png', dpi=200, facecolor='white')
        plt.close()

        # Speed variance
        fig6, ax6 = plt.subplots(figsize=(14, 6))
        window = 50
        if len(self.speed_history) > window:
            variances = [np.std(list(self.speed_history)[max(0, i-window):i+1]) 
                        for i in range(len(self.speed_history))]
            ax6.plot(times, variances, 'purple', linewidth=2)
            ax6.fill_between(times, 0, variances, alpha=0.3, color='purple')
        ax6.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Standard Deviation (m/s)', fontsize=12, fontweight='bold')
        ax6.set_title('Speed Stability', fontsize=14, fontweight='bold', pad=20)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_6_speed_variance.png', dpi=200, facecolor='white')
        plt.close()

        # Speed by state
        fig7, ax7 = plt.subplots(figsize=(10, 6))
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
            bp = ax7.boxplot(data, positions=positions, labels=labels, patch_artist=True, widths=0.6)
            colors = ['red', 'yellow', 'green']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax7.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Traffic Light State', fontsize=12, fontweight='bold')
        ax7.set_title('Speed Distribution by Traffic Light State', fontsize=14, fontweight='bold', pad=20)
        ax7.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('analytics_7_speed_by_state.png', dpi=200, facecolor='white')
        plt.close()

        # Reaction times
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        if self.reaction_times:
            ax8.hist(self.reaction_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
            mean_rt = np.mean(self.reaction_times)
            ax8.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}s')
            ax8.set_xlabel('Reaction Time (seconds)', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax8.set_title('Robot Reaction Time Distribution', fontsize=14, fontweight='bold', pad=20)
            ax8.legend(fontsize=11)
        else:
            ax8.text(0.5, 0.5, 'No reaction time data', transform=ax8.transAxes, fontsize=14, ha='center')
        ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analytics_8_reaction_times.png', dpi=200, facecolor='white')
        plt.close()

        # Summary stats
        fig9, ax9 = plt.subplots(figsize=(10, 8))
        ax9.axis('off')

        stats_text = f"""HYBRID TRAFFIC LIGHT DETECTION - PERFORMANCE REPORT
        ══════════════════════════════════════════════════════
        SPEED CONTROL METRICS
        ══════════════════════════════════════════════════════
        Mean Speed:              {np.mean(self.speed_history):.3f} m/s
        Speed Variance:          {np.var(self.speed_history):.4f}
        Mean Tracking Error:     {np.mean(self.speed_error_history):.3f} m/s
        RMSE:                    {np.sqrt(np.mean(np.array(self.speed_error_history)**2)):.3f} m/s
        Max Speed:               {np.max(self.speed_history):.3f} m/s
        ══════════════════════════════════════════════════════
        HSV DETECTION STATISTICS
        ══════════════════════════════════════════════════════
        Red Light:
        Mean Ratio:            {np.mean(self.red_history):.2f}%
        Std Deviation:         {np.std(self.red_history):.2f}%
        Peak Ratio:            {np.max(self.red_history):.2f}%
        Yellow Light:
        Mean Ratio:            {np.mean(self.yellow_history):.2f}%
        Std Deviation:         {np.std(self.yellow_history):.2f}%
        Peak Ratio:            {np.max(self.yellow_history):.2f}%
        Green Light:
        Mean Ratio:            {np.mean(self.green_history):.2f}%
        Std Deviation:         {np.std(self.green_history):.2f}%
        Peak Ratio:            {np.max(self.green_history):.2f}%
        ══════════════════════════════════════════════════════
        YOLO DETECTION STATISTICS
        ══════════════════════════════════════════════════════
        Red Light:
        Mean Confidence:       {np.mean(self.yolo_red_history):.2f}%
        Std Deviation:         {np.std(self.yolo_red_history):.2f}%
        Peak Confidence:       {np.max(self.yolo_red_history):.2f}%
        Yellow Light:
        Mean Confidence:       {np.mean(self.yolo_yellow_history):.2f}%
        Std Deviation:         {np.std(self.yolo_yellow_history):.2f}%
        Peak Confidence:       {np.max(self.yolo_yellow_history):.2f}%
        Green Light:
        Mean Confidence:       {np.mean(self.yolo_green_history):.2f}%
        Std Deviation:         {np.std(self.yolo_green_history):.2f}%
        Peak Confidence:       {np.max(self.yolo_green_history):.2f}%
        Average Combined Confidence: {np.mean(self.detection_confidence):.2f}
        ══════════════════════════════════════════════════════
        SYSTEM AGREEMENT
        ══════════════════════════════════════════════════════
        Agreement Rate:          {sum(1 for h, y in zip(self.hsv_state_history, self.yolo_state_history) if h == y) / len(self.hsv_state_history) * 100:.1f}%
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
        ax9.text(0.1, 0.95, stats_text, transform=ax9.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        plt.savefig('analytics_9_summary.png', dpi=200, facecolor='white')
        plt.close()

        self.get_logger().info('Analytics saved')

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

if __name__ == 'main': main()


