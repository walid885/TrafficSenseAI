#!/usr/bin/env python3
# traffic_light_robot/camera_debug.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraDebugNode(Node):
    def __init__(self):
        super().__init__('camera_debug')
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.callback, 10)
        
        cv2.namedWindow("Camera Debug", cv2.WINDOW_NORMAL)
        self.get_logger().info('Camera Debug Node - Showing raw feed + HSV analysis')
        
    def callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 30 != 0:
            return
            
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = cv_image.shape[:2]
        
        # ROI extraction
        y1, y2 = int(h * 0.3), int(h * 0.7)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        roi = cv_image[y1:y2, x1:x2]
        
        # HSV analysis
        hsv_full = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Relaxed masks
        red1 = cv2.inRange(hsv_roi, (0, 50, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv_roi, (170, 50, 50), (180, 255, 255))
        red_mask = red1 | red2
        yellow_mask = cv2.inRange(hsv_roi, (15, 50, 50), (35, 255, 255))
        green_mask = cv2.inRange(hsv_roi, (40, 50, 50), (80, 255, 255))
        
        # Calculate percentages
        total = roi.shape[0] * roi.shape[1]
        red_pct = (cv2.countNonZero(red_mask) / total) * 100
        yellow_pct = (cv2.countNonZero(yellow_mask) / total) * 100
        green_pct = (cv2.countNonZero(green_mask) / total) * 100
        
        # Visualization
        canvas = cv_image.copy()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Info overlay
        cv2.putText(canvas, f"RED: {red_pct:.3f}%", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, f"YELLOW: {yellow_pct:.3f}%", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(canvas, f"GREEN: {green_pct:.3f}%", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # HSV histogram for ROI
        h_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        h_max = h_hist.max()
        if h_max > 0:
            h_norm = (h_hist / h_max * 200).astype(np.uint8)
            for i, val in enumerate(h_norm):
                cv2.line(canvas, (i*4, h-int(val[0])), (i*4, h), (255, 255, 255), 1)
        
        cv2.imshow("Camera Debug", canvas)
        cv2.waitKey(1)
        
        self.get_logger().info(
            f'Frame {self.frame_count} | R:{red_pct:.3f}% Y:{yellow_pct:.3f}% G:{green_pct:.3f}%')

def main():
    rclpy.init()
    node = CameraDebugNode()
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