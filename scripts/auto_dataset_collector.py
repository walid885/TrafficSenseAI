#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import signal

class AutoDatasetCollector(Node):
    def __init__(self, output_dir, max_samples, sample_rate):
        super().__init__('auto_dataset_collector')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self.counter = 0
        self.saved_count = 0
        
        # Robot pose (updated from odometry)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Camera intrinsics (adjust for your camera)
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.5
        self.cy = 240.5
        self.img_width = 640
        self.img_height = 480
        
        # Traffic light world positions from SDF
        self.traffic_lights = [
            # GREEN light at x=8m
            {'x': 8, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.15, 'active': True},
            {'x': 8, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.25, 'active': False},
            {'x': 8, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.25, 'active': False},
            
            # YELLOW light at x=28m
            {'x': 28, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.25, 'active': False},
            {'x': 28, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.15, 'active': True},
            {'x': 28, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.12, 'active': False},
            
            # RED light at x=48m
            {'x': 48, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.12, 'active': False},
            {'x': 48, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.12, 'active': False},
            {'x': 48, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.15, 'active': True},
        ]
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.get_logger().info(f'Collector initialized. Target: {max_samples} samples')
        
        # Shutdown handler
        signal.signal(signal.SIGINT, self.shutdown_handler)
    
    def shutdown_handler(self, sig, frame):
        self.get_logger().info(f'Shutting down. Collected {self.saved_count} samples.')
        sys.exit(0)
    
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def project_3d_to_2d(self, x_world, y_world, z_world):
        """Project 3D world point to 2D image"""
        # Transform to robot frame
        dx = x_world - self.robot_x
        dy = y_world - self.robot_y
        
        # Rotate to robot frame
        cos_yaw = np.cos(self.robot_yaw)
        sin_yaw = np.sin(self.robot_yaw)
        x_robot = dx * cos_yaw + dy * sin_yaw
        y_robot = -dx * sin_yaw + dy * cos_yaw
        
        # Camera frame (x=forward, y=left, z=up, camera height=1.0m)
        x_cam = x_robot
        y_cam = y_robot
        z_cam = z_world - 1.0
        
        if x_cam <= 0.1:  # Behind or too close
            return None
        
        # Project to image
        u = self.fx * (-y_cam / x_cam) + self.cx
        v = self.fy * (-z_cam / x_cam) + self.cy
        
        return int(u), int(v), x_cam
    
    def generate_bbox(self, u, v, radius_3d, distance):
        """Generate YOLO bounding box"""
        # Apparent size decreases with distance
        apparent_radius = max(5, int((radius_3d / distance) * self.fx * 1.5))
        
        x1 = max(0, u - apparent_radius)
        y1 = max(0, v - apparent_radius)
        x2 = min(self.img_width - 1, u + apparent_radius)
        y2 = min(self.img_height - 1, v + apparent_radius)
        
        # YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / self.img_width
        y_center = ((y1 + y2) / 2) / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        
        return x_center, y_center, width, height
    
    def image_callback(self, msg):
        if self.saved_count >= self.max_samples:
            self.get_logger().info('Target samples reached. Exiting...')
            raise SystemExit
        
        self.counter += 1
        if self.counter % self.sample_rate != 0:
            return
        
        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Generate annotations
        labels = []
        for tl in self.traffic_lights:
            # Only annotate active lights (emissive in SDF)
            if not tl['active']:
                continue
            
            result = self.project_3d_to_2d(tl['x'], tl['y'], tl['z'])
            if result is None:
                continue
            
            u, v, distance = result
            
            # Check if in frame
            if not (20 < u < self.img_width - 20 and 20 < v < self.img_height - 20):
                continue
            
            # Skip if too far (low resolution)
            if distance > 60:
                continue
            
            x_c, y_c, w, h = self.generate_bbox(u, v, tl['radius'], distance)
            
            # Validate bbox
            if w < 0.01 or h < 0.01 or w > 0.5 or h > 0.5:
                continue
            
            labels.append(f"{tl['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        # Save only if labels exist
        if not labels:
            return
        
        # Save image
        img_filename = f"frame_{self.saved_count:06d}.jpg"
        img_path = os.path.join(self.output_dir, 'images', 'train', img_filename)
        cv2.imwrite(img_path, cv_image)
        
        # Save labels
        label_path = os.path.join(self.output_dir, 'labels', 'train', 
                                  img_filename.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        self.saved_count += 1
        
        if self.saved_count % 50 == 0:
            self.get_logger().info(f'Progress: {self.saved_count}/{self.max_samples}')

def main():
    if len(sys.argv) < 4:
        print("Usage: auto_dataset_collector.py <output_dir> <max_samples> <sample_rate>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    max_samples = int(sys.argv[2])
    sample_rate = int(sys.argv[3])
    
    rclpy.init()
    node = AutoDatasetCollector(output_dir, max_samples, sample_rate)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
