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
import time

class DatasetCollector(Node):
    def __init__(self, output_dir, max_samples, sample_rate):
        super().__init__('dataset_collector')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self.counter = 0
        self.saved_count = 0
        self.last_save_time = time.time()
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False
        
        # Camera intrinsics (common ROS camera defaults)
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.5
        self.cy = 240.5
        self.img_width = 640
        self.img_height = 480
        
        # Traffic lights with CORRECTED positions from SDF
        # Poles at (8,2,0), (28,2,0), (48,2,0) facing robot (rotated 180°)
        # Active lights offset -0.15m in local X (becomes +0.15 after rotation)
        self.traffic_lights = [
            # GREEN at x=8m (active light at z=2.6, radius=0.15)
            {'x': 8.15, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.15, 'active': True},
            {'x': 8.15, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.25, 'active': False},
            {'x': 8.15, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.25, 'active': False},
            
            # YELLOW at x=28m (active light at z=2.8, radius=0.15)
            {'x': 28.15, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.25, 'active': False},
            {'x': 28.15, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.15, 'active': True},
            {'x': 28.15, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.12, 'active': False},
            
            # RED at x=48m (active light at z=3.0, radius=0.15)
            {'x': 48.15, 'y': 2, 'z': 2.6, 'class_id': 2, 'radius': 0.12, 'active': False},
            {'x': 48.15, 'y': 2, 'z': 2.8, 'class_id': 1, 'radius': 0.12, 'active': False},
            {'x': 48.15, 'y': 2, 'z': 3.0, 'class_id': 0, 'radius': 0.15, 'active': True},
        ]
        
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.get_logger().info(f'Collector initialized. Target: {max_samples} samples')
        self.get_logger().info(f'Output: {output_dir}')
        self.get_logger().info('Waiting for data...')
        
        self.status_timer = self.create_timer(5.0, self.print_status)
        signal.signal(signal.SIGINT, self.shutdown_handler)
    
    def print_status(self):
        if not self.odom_received:
            self.get_logger().warn('Waiting for /odom')
        else:
            self.get_logger().info(
                f'Saved: {self.saved_count}/{self.max_samples} | '
                f'Position: ({self.robot_x:.1f}, {self.robot_y:.1f})'
            )
    
    def shutdown_handler(self, sig, frame):
        self.get_logger().info(f'Collected {self.saved_count} samples.')
        sys.exit(0)
    
    def odom_callback(self, msg):
        self.odom_received = True
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def project_3d_to_2d(self, x_world, y_world, z_world):
        """Project 3D world point to 2D image"""
        dx = x_world - self.robot_x
        dy = y_world - self.robot_y
        
        cos_yaw = np.cos(self.robot_yaw)
        sin_yaw = np.sin(self.robot_yaw)
        x_robot = dx * cos_yaw + dy * sin_yaw
        y_robot = -dx * sin_yaw + dy * cos_yaw
        
        # Camera at robot center, height 1.0m
        x_cam = x_robot
        y_cam = y_robot
        z_cam = z_world - 1.0
        
        if x_cam <= 0.1:
            return None
        
        u = self.fx * (-y_cam / x_cam) + self.cx
        v = self.fy * (-z_cam / x_cam) + self.cy
        
        return int(u), int(v), x_cam
    
    def generate_bbox(self, u, v, radius_3d, distance):
        """Generate YOLO bbox"""
        apparent_radius = max(8, int((radius_3d / distance) * self.fx * 2.0))
        
        x1 = max(0, u - apparent_radius)
        y1 = max(0, v - apparent_radius)
        x2 = min(self.img_width - 1, u + apparent_radius)
        y2 = min(self.img_height - 1, v + apparent_radius)
        
        x_center = ((x1 + x2) / 2) / self.img_width
        y_center = ((y1 + y2) / 2) / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        
        return x_center, y_center, width, height
    
    def image_callback(self, msg):
        if self.saved_count >= self.max_samples:
            self.get_logger().info('Target reached. Exiting...')
            raise SystemExit
        
        if not self.odom_received:
            return
        
        self.counter += 1
        if self.counter % self.sample_rate != 0:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return
        
        labels = []
        for tl in self.traffic_lights:
            if not tl['active']:
                continue
            
            result = self.project_3d_to_2d(tl['x'], tl['y'], tl['z'])
            if result is None:
                continue
            
            u, v, distance = result
            
            # Generous frame margins
            if not (10 < u < self.img_width - 10 and 10 < v < self.img_height - 10):
                continue
            
            # Distance range: 2m to 50m
            if distance < 2 or distance > 50:
                continue
            
            x_c, y_c, w, h = self.generate_bbox(u, v, tl['radius'], distance)
            
            if 0.005 < w < 0.6 and 0.005 < h < 0.6:
                labels.append(f"{tl['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        if not labels:
            return
        
        current_time = time.time()
        if current_time - self.last_save_time < 0.05:
            return
        self.last_save_time = current_time
        
        img_filename = f"frame_{self.saved_count:06d}.jpg"
        img_path = os.path.join(self.output_dir, 'images', 'train', img_filename)
        
        try:
            cv2.imwrite(img_path, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            label_path = os.path.join(self.output_dir, 'labels', 'train', 
                                      img_filename.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            self.saved_count += 1
            
            if self.saved_count % 20 == 0:
                self.get_logger().info(f'Progress: {self.saved_count}/{self.max_samples}')
        except Exception as e:
            self.get_logger().error(f'Save failed: {e}')

def main():
    if len(sys.argv) < 4:
        print("Usage: dataset_collector_fixed.py <output_dir> <max_samples> <sample_rate>")
        sys.exit(1)
    
    rclpy.init()
    node = DatasetCollector(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()