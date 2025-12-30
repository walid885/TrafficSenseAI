#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

class RVizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')
        self.bridge = CvBridge()
        self.current_state = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.0
        
        self.red_history = deque(maxlen=1000)
        self.green_history = deque(maxlen=1000)
        self.yellow_history = deque(maxlen=1000)
        self.speed_history = deque(maxlen=1000)
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            String, '/traffic_light_state', self.state_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.marker_pub = self.create_publisher(MarkerArray, '/traffic_visualization', 10)
        self.timer = self.create_timer(0.1, self.publish_markers)
        
        self.get_logger().info('RViz Visualizer started - markers on /traffic_visualization')
        
    def state_callback(self, msg):
        self.current_state = msg.data
        
    def cmd_callback(self, msg):
        self.current_speed = msg.linear.x
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red1 | red2
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        if self.current_state == "GREEN":
            self.target_speed = 0.5
        elif self.current_state == "YELLOW":
            self.target_speed = 0.2
        elif self.current_state == "RED":
            self.target_speed = 0.0
        else:
            self.target_speed = 0.5
        
        self.red_history.append(red_pixels)
        self.green_history.append(green_pixels)
        self.yellow_history.append(yellow_pixels)
        self.speed_history.append(self.current_speed)
        self.frame_count += 1
        
    def publish_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        lifetime = Duration(sec=1, nanosec=0)
        
        # Traffic light sphere
        state_marker = Marker()
        state_marker.header.frame_id = "base_link"
        state_marker.header.stamp = now
        state_marker.ns = "traffic_state"
        state_marker.id = 0
        state_marker.type = Marker.SPHERE
        state_marker.action = Marker.ADD
        state_marker.lifetime = lifetime
        state_marker.pose.position.x = 2.0
        state_marker.pose.position.y = 0.0
        state_marker.pose.position.z = 2.0
        state_marker.pose.orientation.w = 1.0
        state_marker.scale.x = 0.5
        state_marker.scale.y = 0.5
        state_marker.scale.z = 0.5
        
        if self.current_state == "GREEN":
            state_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        elif self.current_state == "YELLOW":
            state_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        elif self.current_state == "RED":
            state_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        else:
            state_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
        
        marker_array.markers.append(state_marker)
        
        # Speed bar
        max_speed = 0.6
        speed_height = max((self.current_speed / max_speed) * 2.0, 0.01)
        
        speed_bar = Marker()
        speed_bar.header.frame_id = "base_link"
        speed_bar.header.stamp = now
        speed_bar.ns = "speed_bar"
        speed_bar.id = 1
        speed_bar.type = Marker.CUBE
        speed_bar.action = Marker.ADD
        speed_bar.lifetime = lifetime
        speed_bar.pose.position.x = 1.5
        speed_bar.pose.position.y = -1.0
        speed_bar.pose.position.z = speed_height / 2.0
        speed_bar.pose.orientation.w = 1.0
        speed_bar.scale.x = 0.2
        speed_bar.scale.y = 0.2
        speed_bar.scale.z = speed_height
        speed_bar.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        marker_array.markers.append(speed_bar)
        
        # Target speed line
        target_line = Marker()
        target_line.header.frame_id = "base_link"
        target_line.header.stamp = now
        target_line.ns = "target_speed"
        target_line.id = 2
        target_line.type = Marker.LINE_STRIP
        target_line.action = Marker.ADD
        target_line.lifetime = lifetime
        target_line.pose.orientation.w = 1.0
        target_line.scale.x = 0.05
        target_line.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        target_height = (self.target_speed / max_speed) * 2.0
        target_line.points = [
            Point(x=1.3, y=-1.0, z=target_height),
            Point(x=1.7, y=-1.0, z=target_height)
        ]
        marker_array.markers.append(target_line)
        
        # Text display
        avg_speed = np.mean(self.speed_history) if len(self.speed_history) > 0 else 0.0
        
        text_marker = Marker()
        text_marker.header.frame_id = "base_link"
        text_marker.header.stamp = now
        text_marker.ns = "text_info"
        text_marker.id = 3
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = lifetime
        text_marker.pose.position.x = 2.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = 2.8
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.3
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = f"{self.current_state}\n{self.current_speed:.2f} m/s\nAvg: {avg_speed:.2f}"
        marker_array.markers.append(text_marker)
        
        # Color intensity bars
        if len(self.red_history) > 0:
            red_val = min(list(self.red_history)[-1] / 5000.0, 1.0)
            green_val = min(list(self.green_history)[-1] / 5000.0, 1.0)
            yellow_val = min(list(self.yellow_history)[-1] / 5000.0, 1.0)
            
            colors_data = [
                (red_val, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7), 0.5),
                (yellow_val, ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7), 0.0),
                (green_val, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7), -0.5)
            ]
            
            for idx, (val, color, y_offset) in enumerate(colors_data):
                bar = Marker()
                bar.header.frame_id = "base_link"
                bar.header.stamp = now
                bar.ns = "color_bars"
                bar.id = 10 + idx
                bar.type = Marker.CUBE
                bar.action = Marker.ADD
                bar.lifetime = lifetime
                bar.pose.position.x = 1.5
                bar.pose.position.y = y_offset
                bar.pose.position.z = max(val, 0.01) / 2.0
                bar.pose.orientation.w = 1.0
                bar.scale.x = 0.15
                bar.scale.y = 0.15
                bar.scale.z = max(val, 0.01)
                bar.color = color
                marker_array.markers.append(bar)
        
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = RVizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()