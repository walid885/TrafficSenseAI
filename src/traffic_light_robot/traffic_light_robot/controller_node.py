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
    """PID Controller for velocity control"""
    def __init__(self, kp, ki, kd, output_min=0.0, output_max=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        
    def compute(self, setpoint, current_value):
        """Compute PID output"""
        current_time = time.time()
        
        if self.prev_time is None:
            self.prev_time = current_time
            dt = 0.1  # Default dt
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.1
        
        # Error calculation
        error = setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Anti-windup clamping
        max_integral = 1.0
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.output_min, min(self.output_max, output))
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        return output, p_term, i_term, d_term
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None


class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        
        # Declare parameters
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.3)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('control_rate', 50.0)  # Hz
        
        # Get parameters
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        control_rate = self.get_parameter('control_rate').value
        
        self.state = State.MOVING
        self.current_light = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.5
        
        # PID controller initialization
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, output_min=0.0, output_max=0.5)
        
        # Speed targets for each state
        self.speed_targets = {
            State.MOVING: 0.5,
            State.SLOWING: 0.2,
            State.STOPPED: 0.0
        }
        
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control loop timer
        control_period = 1.0 / control_rate
        self.timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info(f'Autonomous controller started with PID: Kp={kp}, Ki={ki}, Kd={kd}')
        
    def light_callback(self, msg):
        old_light = self.current_light
        self.current_light = msg.data
        
        if old_light != self.current_light and self.current_light != "UNKNOWN":
            self.get_logger().info(f'Light changed: {old_light} -> {self.current_light}')
            
            # Update state based on light
            if self.current_light == "RED":
                self.state = State.STOPPED
            elif self.current_light == "YELLOW":
                self.state = State.SLOWING
            elif self.current_light == "GREEN":
                self.state = State.MOVING
        
    def control_loop(self):
        cmd = Twist()
        
        # Determine target speed based on state
        self.target_speed = self.speed_targets[self.state]
        
        # PID control
        output, p_term, i_term, d_term = self.pid.compute(self.target_speed, self.current_speed)
        
        # Update current speed (simulated dynamics)
        # In real system, this would come from odometry feedback
        alpha = 0.3  # Smoothing factor for simulated dynamics
        self.current_speed = alpha * output + (1 - alpha) * self.current_speed
        
        # Publish command
        cmd.linear.x = output
        self.cmd_pub.publish(cmd)
        
        # Log PID values periodically (every 1 second)
        if hasattr(self, 'log_counter'):
            self.log_counter += 1
        else:
            self.log_counter = 0
            
        if self.log_counter % 50 == 0:  # Log every 50 iterations (1 sec at 50Hz)
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