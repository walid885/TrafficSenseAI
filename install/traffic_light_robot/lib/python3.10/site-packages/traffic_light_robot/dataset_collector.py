# ~/yolo_traffic_project/scripts/collect_dataset.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        self.bridge = CvBridge()
        self.counter = 0
        self.base_path = os.path.expanduser('~/yolo_traffic_project/dataset/images/train')
        os.makedirs(self.base_path, exist_ok=True)
        
        # Camera intrinsics (adjust for your camera)
        self.fx = 554.25  # Focal length x
        self.fy = 554.25  # Focal length y
        self.cx = 320.5   # Principal point x
        self.cy = 240.5   # Principal point y
        
        # Traffic light positions in world frame
        self.traffic_lights = [
            {'x': 8, 'y': 2, 'z': 2.6, 'state': 'green', 'radius': 0.15},
            {'x': 8, 'y': 2, 'z': 2.8, 'state': 'yellow', 'radius': 0.25},
            {'x': 8, 'y': 2, 'z': 3.0, 'state': 'red', 'radius': 0.25},
            {'x': 28, 'y': 2, 'z': 2.6, 'state': 'green', 'radius': 0.25},
            {'x': 28, 'y': 2, 'z': 2.8, 'state': 'yellow', 'radius': 0.15},
            {'x': 28, 'y': 2, 'z': 3.0, 'state': 'red', 'radius': 0.12},
            {'x': 48, 'y': 2, 'z': 2.6, 'state': 'green', 'radius': 0.12},
            {'x': 48, 'y': 2, 'z': 2.8, 'state': 'yellow', 'radius': 0.12},
            {'x': 48, 'y': 2, 'z': 3.0, 'state': 'red', 'radius': 0.15},
        ]
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.callback, 10)
        
        # Assume robot publishes pose (or use tf2)
        # For simplicity, hardcode robot trajectory
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        self.get_logger().info('Dataset collector started')
    
    def project_3d_to_2d(self, x_world, y_world, z_world):
        """Project 3D world point to 2D image pixel"""
        # Transform world to camera frame (assuming robot at origin facing +x)
        # Camera frame: x=forward, y=left, z=up
        x_cam = x_world - self.robot_x
        y_cam = y_world - self.robot_y
        z_cam = z_world - 1.5  # Camera height ~1.5m
        
        if x_cam <= 0:  # Behind camera
            return None
        
        # Project to image plane
        u = self.fx * (-y_cam / x_cam) + self.cx
        v = self.fy * (-z_cam / x_cam) + self.cy
        
        return int(u), int(v)
    
    def generate_bbox(self, center_x, center_y, radius_3d, distance):
        """Generate bounding box from 3D sphere projection"""
        # Apparent size decreases with distance
        apparent_radius = int((radius_3d / distance) * self.fx)
        x1 = max(0, center_x - apparent_radius)
        y1 = max(0, center_y - apparent_radius)
        x2 = min(639, center_x + apparent_radius)
        y2 = min(479, center_y + apparent_radius)
        return x1, y1, x2, y2
    
    def callback(self, msg):
        if self.counter % 3 != 0:  # Sample every 3rd frame
            self.counter += 1
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = cv_image.shape[:2]
        
        # Save image
        img_filename = f"frame_{self.counter:06d}.jpg"
        img_path = os.path.join(self.base_path, img_filename)
        cv2.imwrite(img_path, cv_image)
        
        # Generate labels
        label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        labels = []
        for tl in self.traffic_lights:
            result = self.project_3d_to_2d(tl['x'], tl['y'], tl['z'])
            if result is None:
                continue
            
            u, v = result
            if not (0 <= u < w and 0 <= v < h):
                continue
            
            distance = np.sqrt((tl['x'] - self.robot_x)**2 + (tl['y'] - self.robot_y)**2)
            x1, y1, x2, y2 = self.generate_bbox(u, v, tl['radius'], distance)
            
            # YOLO format: class x_center y_center width height (normalized)
            class_id = 0 if tl['state'] == 'red' else 1 if tl['state'] == 'yellow' else 2
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
        
        # Save labels
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        self.counter += 1
        if self.counter % 30 == 0:
            self.get_logger().info(f'Collected {self.counter} samples')

def main():
    rclpy.init()
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()