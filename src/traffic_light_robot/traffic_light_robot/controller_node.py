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
            dt = 0.02  # 50Hz control rate
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.02
        
        error = setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        max_integral = 0.5
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = 0.2 * raw_derivative + 0.8 * self.filtered_derivative
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
        
        # PID parameters
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('control_rate', 50.0)
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        control_rate = self.get_parameter('control_rate').value
        
        # State management
        self.state = State.MOVING
        self.current_light = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 0.9  # Start with green assumption
        
        # Speed targets as specified
        self.speed_targets = {
            State.MOVING: 0.9,    # GREEN
            State.SLOWING: 0.7,   # YELLOW
            State.STOPPED: 0.0    # RED
        }
        
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, output_min=0.0, output_max=1.5)
        
        # Subscriptions and publishers
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control loop timer
        control_period = 1.0 / control_rate
        self.timer = self.create_timer(control_period, self.control_loop)
        
        self.log_counter = 0
        
        self.get_logger().info(f'Controller started - Kp={kp}, Ki={ki}, Kd={kd}')
        self.get_logger().info(f'Speed targets: GREEN={self.speed_targets[State.MOVING]}, YELLOW={self.speed_targets[State.SLOWING]}, RED={self.speed_targets[State.STOPPED]}')
        
    def light_callback(self, msg):
        """Handle traffic light state changes"""
        old_light = self.current_light
        new_light = msg.data
        
        # Skip if no change or unknown
        if new_light == old_light:
            return
            
        self.current_light = new_light
        old_state = self.state
        
        # State transitions based on light color
        if new_light == "RED":
            self.state = State.STOPPED
            self.target_speed = self.speed_targets[State.STOPPED]
            
        elif new_light == "YELLOW":
            self.state = State.SLOWING
            self.target_speed = self.speed_targets[State.SLOWING]
            
        elif new_light == "GREEN":
            self.state = State.MOVING
            self.target_speed = self.speed_targets[State.MOVING]
            
        elif new_light == "UNKNOWN":
            # Assume green if unknown
            self.state = State.MOVING
            self.target_speed = self.speed_targets[State.MOVING]
        
        # Log state change
        if old_state != self.state:
            self.get_logger().info(
                f'STATE CHANGE: {old_light} -> {new_light} | '
                f'{old_state.name} -> {self.state.name} | '
                f'Target speed: {self.target_speed:.2f} m/s'
            )
        
    def control_loop(self):
        """Main control loop - runs at 50Hz"""
        # Compute PID output
        output, p_term, i_term, d_term = self.pid.compute(self.target_speed, self.current_speed)
        
        # Smooth speed updates with exponential filter
        alpha = 0.3
        self.current_speed = alpha * output + (1 - alpha) * self.current_speed
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = output
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        
        # Periodic logging (every 1 second = 50 iterations)
        self.log_counter += 1
        if self.log_counter % 50 == 0:
            error = self.target_speed - self.current_speed
            self.get_logger().info(
                f'[{self.state.name}] Light={self.current_light} | '
                f'Target={self.target_speed:.2f} Current={self.current_speed:.2f} | '
                f'Error={error:.3f} | Output={output:.2f} | '
                f'PID[P={p_term:.3f} I={i_term:.3f} D={d_term:.3f}]'
            )

def main():
    rclpy.init()
    node = AutonomousController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutdown
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()