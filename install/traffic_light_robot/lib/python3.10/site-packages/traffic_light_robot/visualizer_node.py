#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficLightVisualizer(Node):
    def __init__(self):
        super().__init__('traffic_light_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        self.state_sub = self.create_subscription(
            String, '/front_camera/image_raw', self.state_callback, 10)
        
        self.get_logger().info('Visualizer started - Press Q to quit')
        
    def state_callback(self, msg):
        self.current_state = msg.data
        
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
        
        # Visualization
        h, w = cv_image.shape[:2]
        
        # Overlay masks
        red_overlay = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        red_overlay[:,:,1:] = 0  # Keep only red channel
        green_overlay = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        green_overlay[:,:,[0,2]] = 0  # Keep only green channel
        yellow_overlay = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        yellow_overlay[:,:,2] = 0  # Remove blue channel
        
        combined = cv2.addWeighted(cv_image, 0.7, red_overlay, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, green_overlay, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, yellow_overlay, 0.3, 0)
        
        # Info panel
        cv2.rectangle(combined, (0, 0), (w, 120), (0, 0, 0), -1)
        
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
        
        # Threshold line
        threshold = 500
        cv2.line(combined, (250, 50), (250 + threshold//5, 50), (255, 255, 255), 2)
        cv2.putText(combined, f"Threshold: {threshold}", 
                    (250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show
        cv2.imshow("Traffic Light Detection", combined)
        cv2.waitKey(1)

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