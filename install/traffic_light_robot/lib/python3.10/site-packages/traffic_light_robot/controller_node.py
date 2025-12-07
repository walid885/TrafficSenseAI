#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from enum import Enum
import time

class State(Enum):
    MOVING = 1
    SLOWING = 2
    STOPPED = 3

class PIDController:
    """PID Controller for velocity control with derivative filtering"""
    def __init__(self, kp, ki, kd, output_min=0.0, output_max=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        self.filtered_derivative = 0.0
        
    def compute(self, setpoint, current_value):
        """Compute PID output with derivative filtering"""
        current_time = time.time()
        
        if self.prev_time is None:
            self.prev_time = current_time
            dt = 0.1
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.1
        
        error = setpoint - current_value
        
        p_term = self.kp * error
        
        self.integral += error * dt
        max_integral = 1.0
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i_term = self.ki * self.integral
        
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = 0.1 * raw_derivative + 0.9 * self.filtered_derivative
        d_term = self.kd * self.filtered_derivative
        
        output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, output))
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output, p_term, i_term, d_term
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        self.filtered_derivative = 0.0


class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        
        # Optimized PID parameters from tuning
        self.declare_parameter('kp', 0.3)
        self.declare_parameter('ki', 0.3)
        self.declare_parameter('kd', 0.03)
        self.declare_parameter('control_rate', 50.0)
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        control_rate = self.get_parameter('control_rate').value
        
        self.state = State.MOVING
        self.current_light = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 1.3
        
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, output_min=0.0, output_max=1.5)
        
        self.speed_targets = {
            State.MOVING: 1.3,
            State.SLOWING: 0.3,
            State.STOPPED: 0.0
        }
        
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        control_period = 1.0 / control_rate
        self.timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info(f'Autonomous controller started with PID: Kp={kp}, Ki={ki}, Kd={kd}')
        
    def light_callback(self, msg):
        old_light = self.current_light
        self.current_light = msg.data
        
        if old_light != self.current_light and self.current_light != "UNKNOWN":
            self.get_logger().info(f'Light changed: {old_light} -> {self.current_light}')
            
            if self.current_light == "RED":
                self.state = State.STOPPED
            elif self.current_light == "YELLOW":
                self.state = State.SLOWING
            elif self.current_light == "GREEN":
                self.state = State.MOVING
        
    def control_loop(self):
        cmd = Twist()
        
        self.target_speed = self.speed_targets[self.state]
        
        output, p_term, i_term, d_term = self.pid.compute(self.target_speed, self.current_speed)
        
        alpha = 0.6
        self.current_speed = alpha * output + (1 - alpha) * self.current_speed
        
        cmd.linear.x = output
        self.cmd_pub.publish(cmd)
        
        if hasattr(self, 'log_counter'):
            self.log_counter += 1
        else:
            self.log_counter = 0
            
        if self.log_counter % 50 == 0:
            self.get_logger().info(
                f'State={self.state.name}, Target={self.target_speed:.2f}, '
                f'Current={self.current_speed:.2f}, Output={output:.2f}, '
                f'P={p_term:.3f}, I={i_term:.3f}, D={d_term:.3f}'
            )

def main():
    rclpy.init()
    node = AutonomousController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()