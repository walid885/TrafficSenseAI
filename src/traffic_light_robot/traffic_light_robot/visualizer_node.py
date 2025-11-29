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
from ultralytics import YOLO

class YOLOTrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('yolo_traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        
        # Load YOLO model
        # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
        # You can also use a custom trained model for traffic lights
        self.model = YOLO('yolov8n.pt')  # Change to your model path
        self.get_logger().info('YOLO model loaded')
        
        # Traffic light class names (update based on your model)
        self.traffic_light_classes = ['red_light', 'yellow_light', 'green_light', 'traffic_light']
        
        # Data logging
        self.red_detections = deque(maxlen=1000)
        self.green_detections = deque(maxlen=1000)
        self.yellow_detections = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.frame_count = 0
        
        # Detection parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        
        self.get_logger().info('YOLO Visualizer started - Press Q to quit and generate plot')
        
    def state_callback(self, msg):
        self.current_state = msg.data
        
    def determine_light_color(self, box, image):
        """Determine traffic light color from bounding box region using HSV analysis"""
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color masks
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # Calculate percentages
        red_pct = (red_pixels / total_pixels) * 100
        green_pct = (green_pixels / total_pixels) * 100
        yellow_pct = (yellow_pixels / total_pixels) * 100
        
        max_pct = max(red_pct, green_pct, yellow_pct)
        
        if max_pct < 1.0:  # Threshold for valid detection
            return None, 0
        
        if red_pct == max_pct:
            return "RED", red_pixels
        elif green_pct == max_pct:
            return "GREEN", green_pixels
        elif yellow_pct == max_pct:
            return "YELLOW", yellow_pixels
        
        return None, 0
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run YOLO inference
        results = self.model(cv_image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Initialize detection counts
        red_count = 0
        green_count = 0
        yellow_count = 0
        max_confidence = 0.0
        detected_state = "UNKNOWN"
        
        # Process detections
        annotated_frame = cv_image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Determine color from the detected region
                color_state, pixel_count = self.determine_light_color(
                    [x1, y1, x2, y2], cv_image)
                
                if color_state:
                    if color_state == "RED":
                        red_count += 1
                        box_color = (0, 0, 255)
                    elif color_state == "GREEN":
                        green_count += 1
                        box_color = (0, 255, 0)
                    elif color_state == "YELLOW":
                        yellow_count += 1
                        box_color = (0, 255, 255)
                    
                    # Update detected state based on highest confidence
                    if conf > max_confidence:
                        max_confidence = conf
                        detected_state = color_state
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, 
                                (int(x1), int(y1)), (int(x2), int(y2)), 
                                box_color, 2)
                    
                    # Draw label
                    label = f"{color_state} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame,
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)),
                                box_color, -1)
                    cv2.putText(annotated_frame, label,
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Log data
        self.red_detections.append(red_count)
        self.green_detections.append(green_count)
        self.yellow_detections.append(yellow_count)
        self.confidence_history.append(max_confidence)
        self.timestamps.append(self.frame_count)
        self.frame_count += 1
        
        # Update current state if detection confidence is high
        if max_confidence > 0.5:
            self.current_state = detected_state
        
        # Create visualization
        h, w = annotated_frame.shape[:2]
        
        # Info panel
        cv2.rectangle(annotated_frame, (0, 0), (w, 150), (0, 0, 0), -1)
        
        # State
        color = (0, 255, 0) if self.current_state == "GREEN" else \
                (0, 255, 255) if self.current_state == "YELLOW" else \
                (0, 0, 255) if self.current_state == "RED" else (128, 128, 128)
        cv2.putText(annotated_frame, f"STATE: {self.current_state}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Detection counts
        cv2.putText(annotated_frame, f"RED Lights: {red_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"YELLOW Lights: {yellow_count}", 
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"GREEN Lights: {green_count}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Confidence
        cv2.putText(annotated_frame, f"Confidence: {max_confidence:.2f}", 
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS counter
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", 
                    (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show
        cv2.imshow("YOLO Traffic Light Detection", annotated_frame)
        key = cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            self.generate_plot()
            raise KeyboardInterrupt
    
    def generate_plot(self):
        if len(self.timestamps) == 0:
            self.get_logger().warn('No data to plot')
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time series plot - Detection counts
        ax1.plot(list(self.timestamps), list(self.red_detections), 'r-', 
                label='Red', linewidth=1.5)
        ax1.plot(list(self.timestamps), list(self.green_detections), 'g-', 
                label='Green', linewidth=1.5)
        ax1.plot(list(self.timestamps), list(self.yellow_detections), 'y-', 
                label='Yellow', linewidth=1.5)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Detection Count')
        ax1.set_title('Traffic Light Detections Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f0f0f0')
        
        # Confidence over time
        ax2.plot(list(self.timestamps), list(self.confidence_history), 'b-', 
                linewidth=1.5)
        ax2.axhline(y=0.5, color='red', linestyle='--', 
                   label='Threshold (0.5)', linewidth=2)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Detection Confidence Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f0f0f0')
        ax2.set_ylim([0, 1])
        
        # Histogram/Distribution
        ax3.hist([list(self.red_detections), list(self.green_detections), 
                 list(self.yellow_detections)], 
                bins=range(0, max(max(self.red_detections), 
                                 max(self.green_detections), 
                                 max(self.yellow_detections)) + 2),
                label=['Red', 'Green', 'Yellow'], 
                color=['red', 'green', 'yellow'], alpha=0.7)
        ax3.set_xlabel('Detection Count per Frame')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Detection Count Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#f0f0f0')
        
        # Detection statistics pie chart
        total_red = sum(self.red_detections)
        total_green = sum(self.green_detections)
        total_yellow = sum(self.yellow_detections)
        
        if total_red + total_green + total_yellow > 0:
            sizes = [total_red, total_green, total_yellow]
            colors_pie = ['red', 'green', 'yellow']
            labels_pie = ['Red', 'Green', 'Yellow']
            ax4.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90)
            ax4.set_title('Total Detections Distribution')
        else:
            ax4.text(0.5, 0.5, 'No detections', ha='center', va='center')
            ax4.set_title('Total Detections Distribution')
        
        # Statistics
        stats_text = f"""YOLO Detection Statistics:
Red    - Total: {total_red}, Avg: {np.mean(self.red_detections):.2f}, Max: {np.max(self.red_detections)}
Green  - Total: {total_green}, Avg: {np.mean(self.green_detections):.2f}, Max: {np.max(self.green_detections)}
Yellow - Total: {total_yellow}, Avg: {np.mean(self.yellow_detections):.2f}, Max: {np.max(self.yellow_detections)}
Confidence - Mean: {np.mean(self.confidence_history):.3f}, Max: {np.max(self.confidence_history):.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('yolo_traffic_light_analysis.png', dpi=150, facecolor='white')
        self.get_logger().info('Plot saved: yolo_traffic_light_analysis.png')
        plt.show()

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