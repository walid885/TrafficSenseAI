#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class AutoRobotDriver(Node):
    def __init__(self, duration):
        super().__init__('auto_robot_driver')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.duration = duration
        self.start_time = time.time()
        
        self.timer = self.create_timer(0.1, self.drive_callback)
        self.get_logger().info(f'Autonomous driver started for {duration}s')
    
    def drive_callback(self):
        elapsed = time.time() - self.start_time
        
        if elapsed > self.duration:
            self.get_logger().info('Drive complete')
            raise SystemExit
        
        msg = Twist()
        
        # Simple forward motion with slight variations
        if elapsed < 10:
            msg.linear.x = 0.3
        elif elapsed < 15:
            msg.linear.x = 0.5
        elif elapsed < 20:
            msg.linear.x = 0.2
        else:
            # Repeat pattern
            phase = (elapsed - 20) % 15
            if phase < 5:
                msg.linear.x = 0.4
            elif phase < 10:
                msg.linear.x = 0.3
            else:
                msg.linear.x = 0.5
        
        self.pub.publish(msg)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: auto_robot_driver.py <duration_seconds>")
        sys.exit(1)
    
    duration = int(sys.argv[1])
    
    rclpy.init()
    node = AutoRobotDriver(duration)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Stop robot
        msg = Twist()
        node.pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
