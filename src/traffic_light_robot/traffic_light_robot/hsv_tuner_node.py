#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVTuner(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        # ROI parameters
        self.roi_y_start = 0.2
        self.roi_y_end = 0.6
        self.roi_x_start = 0.3
        self.roi_x_end = 0.7
        
        # Current color mode
        self.color_mode = 0  # 0=Red, 1=Yellow, 2=Green
        
        # HSV ranges for each color
        self.red_lower = np.array([0, 80, 70])
        self.red_upper = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 80, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.yellow_lower = np.array([15, 80, 70])
        self.yellow_upper = np.array([35, 255, 255])
        
        self.green_lower = np.array([35, 80, 70])
        self.green_upper = np.array([85, 255, 255])
        
        # Preprocessing params
        self.blur_kernel = 5
        self.clahe_clip = 3.0
        self.morph_kernel = 5
        
        # Setup windows
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask Output", cv2.WINDOW_NORMAL)
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        
        # Create trackbars
        self.create_trackbars()
        
        self.get_logger().info('HSV Tuner started - Press R/Y/G to switch colors, Q to quit')
        
    def create_trackbars(self):
        # ROI trackbars
        cv2.createTrackbar('ROI Y Start', 'HSV Tuner', int(self.roi_y_start * 100), 100, self.on_roi_change)
        cv2.createTrackbar('ROI Y End', 'HSV Tuner', int(self.roi_y_end * 100), 100, self.on_roi_change)
        cv2.createTrackbar('ROI X Start', 'HSV Tuner', int(self.roi_x_start * 100), 100, self.on_roi_change)
        cv2.createTrackbar('ROI X End', 'HSV Tuner', int(self.roi_x_end * 100), 100, self.on_roi_change)
        
        # Color selection
        cv2.createTrackbar('Color (0=R 1=Y 2=G)', 'HSV Tuner', 0, 2, self.on_color_change)
        
        # HSV trackbars
        cv2.createTrackbar('H Lower', 'HSV Tuner', 0, 180, lambda x: None)
        cv2.createTrackbar('H Upper', 'HSV Tuner', 10, 180, lambda x: None)
        cv2.createTrackbar('S Lower', 'HSV Tuner', 80, 255, lambda x: None)
        cv2.createTrackbar('S Upper', 'HSV Tuner', 255, 255, lambda x: None)
        cv2.createTrackbar('V Lower', 'HSV Tuner', 70, 255, lambda x: None)
        cv2.createTrackbar('V Upper', 'HSV Tuner', 255, 255, lambda x: None)
        
        # Preprocessing trackbars
        cv2.createTrackbar('Blur Kernel', 'HSV Tuner', 5, 15, self.on_blur_change)
        cv2.createTrackbar('CLAHE Clip x10', 'HSV Tuner', 30, 100, self.on_clahe_change)
        cv2.createTrackbar('Morph Kernel', 'HSV Tuner', 5, 15, self.on_morph_change)
        
        self.update_trackbars_for_color()
    
    def on_roi_change(self, val):
        self.roi_y_start = cv2.getTrackbarPos('ROI Y Start', 'HSV Tuner') / 100.0
        self.roi_y_end = cv2.getTrackbarPos('ROI Y End', 'HSV Tuner') / 100.0
        self.roi_x_start = cv2.getTrackbarPos('ROI X Start', 'HSV Tuner') / 100.0
        self.roi_x_end = cv2.getTrackbarPos('ROI X End', 'HSV Tuner') / 100.0
    
    def on_color_change(self, val):
        self.color_mode = val
        self.update_trackbars_for_color()
    
    def on_blur_change(self, val):
        self.blur_kernel = val if val % 2 == 1 else val + 1
    
    def on_clahe_change(self, val):
        self.clahe_clip = val / 10.0
    
    def on_morph_change(self, val):
        self.morph_kernel = val if val % 2 == 1 else val + 1
    
    def update_trackbars_for_color(self):
        if self.color_mode == 0:  # Red
            cv2.setTrackbarPos('H Lower', 'HSV Tuner', self.red_lower[0])
            cv2.setTrackbarPos('H Upper', 'HSV Tuner', self.red_upper[0])
            cv2.setTrackbarPos('S Lower', 'HSV Tuner', self.red_lower[1])
            cv2.setTrackbarPos('S Upper', 'HSV Tuner', self.red_upper[1])
            cv2.setTrackbarPos('V Lower', 'HSV Tuner', self.red_lower[2])
            cv2.setTrackbarPos('V Upper', 'HSV Tuner', self.red_upper[2])
        elif self.color_mode == 1:  # Yellow
            cv2.setTrackbarPos('H Lower', 'HSV Tuner', self.yellow_lower[0])
            cv2.setTrackbarPos('H Upper', 'HSV Tuner', self.yellow_upper[0])
            cv2.setTrackbarPos('S Lower', 'HSV Tuner', self.yellow_lower[1])
            cv2.setTrackbarPos('S Upper', 'HSV Tuner', self.yellow_upper[1])
            cv2.setTrackbarPos('V Lower', 'HSV Tuner', self.yellow_lower[2])
            cv2.setTrackbarPos('V Upper', 'HSV Tuner', self.yellow_upper[2])
        elif self.color_mode == 2:  # Green
            cv2.setTrackbarPos('H Lower', 'HSV Tuner', self.green_lower[0])
            cv2.setTrackbarPos('H Upper', 'HSV Tuner', self.green_upper[0])
            cv2.setTrackbarPos('S Lower', 'HSV Tuner', self.green_lower[1])
            cv2.setTrackbarPos('S Upper', 'HSV Tuner', self.green_upper[1])
            cv2.setTrackbarPos('V Lower', 'HSV Tuner', self.green_lower[2])
            cv2.setTrackbarPos('V Upper', 'HSV Tuner', self.green_upper[2])
    
    def save_current_values(self):
        h_l = cv2.getTrackbarPos('H Lower', 'HSV Tuner')
        h_u = cv2.getTrackbarPos('H Upper', 'HSV Tuner')
        s_l = cv2.getTrackbarPos('S Lower', 'HSV Tuner')
        s_u = cv2.getTrackbarPos('S Upper', 'HSV Tuner')
        v_l = cv2.getTrackbarPos('V Lower', 'HSV Tuner')
        v_u = cv2.getTrackbarPos('V Upper', 'HSV Tuner')
        
        if self.color_mode == 0:
            self.red_lower = np.array([h_l, s_l, v_l])
            self.red_upper = np.array([h_u, s_u, v_u])
        elif self.color_mode == 1:
            self.yellow_lower = np.array([h_l, s_l, v_l])
            self.yellow_upper = np.array([h_u, s_u, v_u])
        elif self.color_mode == 2:
            self.green_lower = np.array([h_l, s_l, v_l])
            self.green_upper = np.array([h_u, s_u, v_u])
    
    def get_roi(self, img):
        h, w = img.shape[:2]
        y_start = int(h * self.roi_y_start)
        y_end = int(h * self.roi_y_end)
        x_start = int(w * self.roi_x_start)
        x_end = int(w * self.roi_x_end)
        return img[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)
    
    def preprocess(self, img):
        if self.blur_kernel > 0:
            blurred = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)
        else:
            blurred = img
        
        if self.clahe_clip > 0:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        return blurred
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Draw ROI on full image
            h, w = cv_image.shape[:2]
            y_start = int(h * self.roi_y_start)
            y_end = int(h * self.roi_y_end)
            x_start = int(w * self.roi_x_start)
            x_end = int(w * self.roi_x_end)
            
            full_display = cv_image.copy()
            cv2.rectangle(full_display, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
            
            # Get ROI and preprocess
            roi, roi_coords = self.get_roi(cv_image)
            processed = self.preprocess(roi)
            
            # Convert to HSV
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            
            # Get trackbar values
            h_l = cv2.getTrackbarPos('H Lower', 'HSV Tuner')
            h_u = cv2.getTrackbarPos('H Upper', 'HSV Tuner')
            s_l = cv2.getTrackbarPos('S Lower', 'HSV Tuner')
            s_u = cv2.getTrackbarPos('S Upper', 'HSV Tuner')
            v_l = cv2.getTrackbarPos('V Lower', 'HSV Tuner')
            v_u = cv2.getTrackbarPos('V Upper', 'HSV Tuner')
            
            # Create mask based on current color
            if self.color_mode == 0:  # Red
                mask1 = cv2.inRange(hsv, np.array([h_l, s_l, v_l]), np.array([h_u, s_u, v_u]))
                mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
                mask = mask1 | mask2
                color_name = "RED"
                color_bgr = (0, 0, 255)
            elif self.color_mode == 1:  # Yellow
                mask = cv2.inRange(hsv, np.array([h_l, s_l, v_l]), np.array([h_u, s_u, v_u]))
                color_name = "YELLOW"
                color_bgr = (0, 255, 255)
            else:  # Green
                mask = cv2.inRange(hsv, np.array([h_l, s_l, v_l]), np.array([h_u, s_u, v_u]))
                color_name = "GREEN"
                color_bgr = (0, 255, 0)
            
            # Apply morphology
            if self.morph_kernel > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate statistics
            pixel_count = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            percentage = (pixel_count / total_pixels) * 100
            
            # Create mask visualization
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(processed, 0.6, mask_color, 0.4, 0)
            
            # Add text to displays
            info_text = [
                f"Mode: {color_name}",
                f"HSV: [{h_l},{s_l},{v_l}] - [{h_u},{s_u},{v_u}]",
                f"Pixels: {pixel_count} ({percentage:.2f}%)",
                f"Blur: {self.blur_kernel} | CLAHE: {self.clahe_clip:.1f} | Morph: {self.morph_kernel}",
                "",
                "Controls:",
                "R = Red | Y = Yellow | G = Green",
                "S = Save values | P = Print config | Q = Quit"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(full_display, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                y_offset += 25
            
            # Show windows
            cv2.imshow("HSV Tuner", full_display)
            cv2.imshow("ROI", processed)
            cv2.imshow("Mask Output", result)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                self.print_config()
                raise KeyboardInterrupt
            elif key == ord('r') or key == ord('R'):
                self.color_mode = 0
                cv2.setTrackbarPos('Color (0=R 1=Y 2=G)', 'HSV Tuner', 0)
                self.update_trackbars_for_color()
            elif key == ord('y') or key == ord('Y'):
                self.color_mode = 1
                cv2.setTrackbarPos('Color (0=R 1=Y 2=G)', 'HSV Tuner', 1)
                self.update_trackbars_for_color()
            elif key == ord('g') or key == ord('G'):
                self.color_mode = 2
                cv2.setTrackbarPos('Color (0=R 1=Y 2=G)', 'HSV Tuner', 2)
                self.update_trackbars_for_color()
            elif key == ord('s') or key == ord('S'):
                self.save_current_values()
                self.get_logger().info(f'{color_name} values saved')
            elif key == ord('p') or key == ord('P'):
                self.print_config()
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
    
    def print_config(self):
        print("\n" + "="*80)
        print("OPTIMIZED HSV CONFIGURATION")
        print("="*80)
        print("\nCopy these values to your detector node:\n")
        print("# Red")
        print(f"red1 = cv2.inRange(hsv, {tuple(self.red_lower)}, {tuple(self.red_upper)})")
        print(f"red2 = cv2.inRange(hsv, {tuple(self.red_lower2)}, {tuple(self.red_upper2)})")
        print("red_mask = red1 | red2")
        print("\n# Yellow")
        print(f"yellow_mask = cv2.inRange(hsv, {tuple(self.yellow_lower)}, {tuple(self.yellow_upper)})")
        print("\n# Green")
        print(f"green_mask = cv2.inRange(hsv, {tuple(self.green_lower)}, {tuple(self.green_upper)})")
        print("\n# ROI Parameters")
        print(f"roi_y_start = {self.roi_y_start}")
        print(f"roi_y_end = {self.roi_y_end}")
        print(f"roi_x_start = {self.roi_x_start}")
        print(f"roi_x_end = {self.roi_x_end}")
        print("\n# Preprocessing")
        print(f"blur_kernel = {self.blur_kernel}")
        print(f"clahe_clip = {self.clahe_clip}")
        print(f"morph_kernel = {self.morph_kernel}")
        print("="*80 + "\n")


def main():
    rclpy.init()
    node = HSVTuner()
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