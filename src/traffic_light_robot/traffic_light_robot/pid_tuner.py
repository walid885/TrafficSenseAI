#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from enum import Enum
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import sys

class State(Enum):
    MOVING = 1
    SLOWING = 2
    STOPPED = 3

class PIDController:
    def __init__(self, kp, ki, kd, output_min=0.0, output_max=1.0):
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
        self.integral = max(-1.0, min(1.0, self.integral))
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
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        self.filtered_derivative = 0.0


class PIDTuner(Node):
    def __init__(self, kp, ki, kd, test_duration=30.0, iteration=0):
        super().__init__('pid_tuner')
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.iteration = iteration
        self.test_duration = test_duration
        self.start_time = time.time()
        
        self.state = State.MOVING
        self.current_light = "UNKNOWN"
        self.current_speed = 0.0
        self.target_speed = 1.0
        
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, output_min=0.0, output_max=1.0)
        
        self.speed_targets = {
            State.MOVING: 1.0,
            State.SLOWING: 0.2,
            State.STOPPED: 0.0
        }
        
        # Data logging
        self.timestamps = deque()
        self.speed_history = deque()
        self.target_history = deque()
        self.error_history = deque()
        self.output_history = deque()
        self.p_history = deque()
        self.i_history = deque()
        self.d_history = deque()
        self.state_history = deque()
        
        self.light_sub = self.create_subscription(
            String, '/traffic_light_state', self.light_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.timer = self.create_timer(0.02, self.control_loop)  # 50Hz
        
        self.get_logger().info(f'Tuning [{iteration}]: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}')
        
    def light_callback(self, msg):
        old_light = self.current_light
        self.current_light = msg.data
        
        if old_light != self.current_light and self.current_light != "UNKNOWN":
            if self.current_light == "RED":
                self.state = State.STOPPED
            elif self.current_light == "YELLOW":
                self.state = State.SLOWING
            elif self.current_light == "GREEN":
                self.state = State.MOVING
        
    def control_loop(self):
        elapsed = time.time() - self.start_time
        
        if elapsed > self.test_duration:
            self.finalize()
            raise KeyboardInterrupt
        
        cmd = Twist()
        self.target_speed = self.speed_targets[self.state]
        
        output, p_term, i_term, d_term = self.pid.compute(self.target_speed, self.current_speed)
        
        alpha = 0.6
        self.current_speed = alpha * output + (1 - alpha) * self.current_speed
        
        cmd.linear.x = output
        self.cmd_pub.publish(cmd)
        
        # Log data
        self.timestamps.append(elapsed)
        self.speed_history.append(self.current_speed)
        self.target_history.append(self.target_speed)
        self.error_history.append(abs(self.target_speed - self.current_speed))
        self.output_history.append(output)
        self.p_history.append(p_term)
        self.i_history.append(i_term)
        self.d_history.append(d_term)
        self.state_history.append(self.state.name)
    
    def finalize(self):
        self.get_logger().info('Test complete. Generating plots...')
        
        # Calculate metrics
        errors = np.array(self.error_history)
        speeds = np.array(self.speed_history)
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        speed_variance = np.var(speeds)
        overshoot = np.max(speeds) - 1.0 if np.max(speeds) > 1.0 else 0.0
        
        # Settling time (time to stay within 5% of target)
        settling_time = None
        target_band = 0.05
        for i in range(len(self.error_history) - 50):
            if all(e < target_band for e in list(self.error_history)[i:i+50]):
                settling_time = self.timestamps[i]
                break
        
        # Generate plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        times = list(self.timestamps)
        
        # 1. Speed tracking
        axes[0, 0].plot(times, self.speed_history, 'b-', label='Actual', linewidth=2)
        axes[0, 0].plot(times, self.target_history, 'r--', label='Target', linewidth=2)
        axes[0, 0].fill_between(times, self.speed_history, self.target_history, alpha=0.3)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed (m/s)')
        axes[0, 0].set_title('Speed Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Tracking error
        axes[0, 1].plot(times, self.error_history, 'orange', linewidth=2)
        axes[0, 1].axhline(y=0.05, color='green', linestyle='--', label='5% band')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (m/s)')
        axes[0, 1].set_title('Tracking Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PID components
        axes[1, 0].plot(times, self.p_history, 'r-', label='P', alpha=0.7)
        axes[1, 0].plot(times, self.i_history, 'g-', label='I', alpha=0.7)
        axes[1, 0].plot(times, self.d_history, 'b-', label='D', alpha=0.7)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Term Value')
        axes[1, 0].set_title('PID Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Control output
        axes[1, 1].plot(times, self.output_history, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Output')
        axes[1, 1].set_title('Control Output')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Speed variance (rolling window)
        window = 50
        variances = []
        for i in range(len(speeds)):
            window_data = speeds[max(0, i-window):i+1]
            variances.append(np.std(window_data))
        axes[2, 0].plot(times, variances, 'cyan', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Std Dev')
        axes[2, 0].set_title('Rolling Speed Variance')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Metrics summary
        axes[2, 1].axis('off')
        metrics_text = f"""PID TUNING RESULTS
        
Parameters:
  Kp = {self.kp:.3f}
  Ki = {self.ki:.3f}
  Kd = {self.kd:.3f}

Performance Metrics:
  MAE:           {mae:.4f} m/s
  RMSE:          {rmse:.4f} m/s
  Max Error:     {max_error:.4f} m/s
  Speed Var:     {speed_variance:.6f}
  Overshoot:     {overshoot:.4f} m/s
  Settling Time: {settling_time:.2f}s if settling_time else 'N/A'
  
Score: {self.calculate_score(mae, rmse, speed_variance, overshoot, settling_time):.2f}
"""
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'PID Tuning Iteration {self.iteration}: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'tuning_results/iteration_{self.iteration:03d}_kp{self.kp:.3f}_ki{self.ki:.3f}_kd{self.kd:.3f}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        results = {
            'iteration': self.iteration,
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'mae': float(mae),
            'rmse': float(rmse),
            'max_error': float(max_error),
            'speed_variance': float(speed_variance),
            'overshoot': float(overshoot),
            'settling_time': float(settling_time) if settling_time else None,
            'score': self.calculate_score(mae, rmse, speed_variance, overshoot, settling_time)
        }
        
        json_file = f'tuning_results/iteration_{self.iteration:03d}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.get_logger().info(f'Results saved: {filename}, {json_file}')
    
    def calculate_score(self, mae, rmse, variance, overshoot, settling_time):
        """Lower score is better"""
        score = (
            10.0 * mae +           # Penalize average error
            15.0 * rmse +          # Penalize RMS error heavily
            50.0 * variance +      # Penalize variance heavily
            20.0 * overshoot +     # Penalize overshoot
            0.1 * (settling_time if settling_time else 30.0)  # Penalize slow settling
        )
        return score


def main():
    if len(sys.argv) != 5:
        print("Usage: pid_tuner.py <kp> <ki> <kd> <iteration>")
        sys.exit(1)
    
    kp = float(sys.argv[1])
    ki = float(sys.argv[2])
    kd = float(sys.argv[3])
    iteration = int(sys.argv[4])
    
    rclpy.init()
    node = PIDTuner(kp, ki, kd, test_duration=30.0, iteration=iteration)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()