#!/usr/bin/env python3
"""
PID Performance Analysis and Visualization Tool
Generates comprehensive graphs after autonomous controller shutdown
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import sys
from datetime import datetime

class PIDAnalyzer:
    def __init__(self, data_file='pid_performance.json'):
        """Initialize analyzer with performance data"""
        try:
            with open(data_file, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Loaded performance data from {data_file}")
            print(f"  Data points: {len(self.data['time'])}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find {data_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON in {data_file}")
            sys.exit(1)
            
    def calculate_metrics(self):
        """Calculate performance metrics"""
        errors = np.array(self.data['error'])
        velocity = np.array(self.data['velocity'])
        setpoint = np.array(self.data['setpoint'])
        
        metrics = {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(np.abs(errors)),
            'steady_state_error': np.mean(np.abs(errors[-20:])),
            'avg_velocity': np.mean(velocity),
            'settling_time': self._calculate_settling_time(),
            'overshoot': self._calculate_overshoot(),
            'response_time': self._calculate_response_time()
        }
        
        return metrics
    
    def _calculate_settling_time(self):
        """Calculate time to settle within 5% of target"""
        errors = np.array(self.data['error'])
        setpoints = np.array(self.data['setpoint'])
        time = np.array(self.data['time'])
        
        # Find state transitions
        transitions = np.where(np.diff(setpoints) != 0)[0]
        
        if len(transitions) == 0:
            return None
            
        settling_times = []
        for trans_idx in transitions:
            if trans_idx + 1 >= len(setpoints):
                continue
                
            target = setpoints[trans_idx + 1]
            threshold = 0.05 * abs(target) if target != 0 else 0.025
            
            # Find when error stays within threshold
            for i in range(trans_idx + 1, len(errors)):
                if abs(errors[i]) <= threshold:
                    # Check if it stays settled for next 10 samples
                    if i + 10 < len(errors):
                        if all(abs(errors[i:i+10]) <= threshold):
                            settling_times.append(time[i] - time[trans_idx])
                            break
        
        return np.mean(settling_times) if settling_times else None
    
    def _calculate_overshoot(self):
        """Calculate maximum overshoot percentage"""
        velocity = np.array(self.data['velocity'])
        setpoint = np.array(self.data['setpoint'])
        
        # Find state transitions
        transitions = np.where(np.diff(setpoint) != 0)[0]
        
        if len(transitions) == 0:
            return 0.0
            
        overshoots = []
        for trans_idx in transitions:
            if trans_idx + 1 >= len(setpoint):
                continue
                
            target = setpoint[trans_idx + 1]
            if target == 0:
                continue
                
            # Look at next 50 samples after transition
            window = velocity[trans_idx:min(trans_idx+50, len(velocity))]
            max_val = np.max(window)
            
            if max_val > target:
                overshoot = ((max_val - target) / target) * 100
                overshoots.append(overshoot)
        
        return np.max(overshoots) if overshoots else 0.0
    
    def _calculate_response_time(self):
        """Calculate time to reach 90% of target after transition"""
        velocity = np.array(self.data['velocity'])
        setpoint = np.array(self.data['setpoint'])
        time = np.array(self.data['time'])
        
        transitions = np.where(np.diff(setpoint) != 0)[0]
        
        if len(transitions) == 0:
            return None
            
        response_times = []
        for trans_idx in transitions:
            if trans_idx + 1 >= len(setpoint):
                continue
                
            target = setpoint[trans_idx + 1]
            threshold = 0.9 * target
            
            for i in range(trans_idx + 1, len(velocity)):
                if velocity[i] >= threshold:
                    response_times.append(time[i] - time[trans_idx])
                    break
        
        return np.mean(response_times) if response_times else None
    
    def plot_comprehensive_analysis(self, output_file='pid_performance_analysis.png'):
        """Generate comprehensive analysis plots"""
        
        metrics = self.calculate_metrics()
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('PID Autonomous Controller - Performance Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Extract data
        time = np.array(self.data['time'])
        velocity = np.array(self.data['velocity'])
        setpoint = np.array(self.data['setpoint'])
        error = np.array(self.data['error'])
        p_term = np.array(self.data['p_term'])
        i_term = np.array(self.data['i_term'])
        d_term = np.array(self.data['d_term'])
        output = np.array(self.data['output'])
        state = np.array(self.data['state'])
        
        # 1. Velocity Tracking (Main plot - spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, setpoint, 'r--', linewidth=2, label='Target Velocity', alpha=0.8)
        ax1.plot(time, velocity, 'b-', linewidth=1.5, label='Actual Velocity', alpha=0.9)
        ax1.fill_between(time, velocity, setpoint, alpha=0.2, color='gray')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Velocity (m/s)', fontsize=11)
        ax1.set_title('Velocity Tracking Performance', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add state regions
        self._add_state_regions(ax1, time, state)
        
        # 2. Tracking Error
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time, error, 'r-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(time, 0, error, alpha=0.3, color='red')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Error (m/s)', fontsize=11)
        ax2.set_title('Tracking Error', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. PID Terms Decomposition
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(time, p_term, 'r-', linewidth=1.5, label='P Term', alpha=0.8)
        ax3.plot(time, i_term, 'g-', linewidth=1.5, label='I Term', alpha=0.8)
        ax3.plot(time, d_term, 'b-', linewidth=1.5, label='D Term', alpha=0.8)
        ax3.plot(time, output, 'k-', linewidth=2, label='Total Output', alpha=0.9)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Control Signal', fontsize=11)
        ax3.set_title('PID Terms Contribution Analysis', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10, ncol=4)
        ax3.grid(True, alpha=0.3)
        
        # 4. Control Output
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(time, output, 'purple', linewidth=1.5)
        ax4.fill_between(time, 0, output, alpha=0.3, color='purple')
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Output (m/s)', fontsize=11)
        ax4.set_title('Control Output Signal', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Error Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(error, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Error (m/s)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Error Distribution', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Phase Plot (Error vs Error Rate)
        ax6 = fig.add_subplot(gs[2, 2])
        error_rate = np.gradient(error, time)
        ax6.plot(error, error_rate, 'b-', linewidth=1, alpha=0.6)
        ax6.scatter(error[0], error_rate[0], c='green', s=100, marker='o', 
                   label='Start', zorder=5)
        ax6.scatter(error[-1], error_rate[-1], c='red', s=100, marker='X', 
                   label='End', zorder=5)
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax6.set_xlabel('Error (m/s)', fontsize=11)
        ax6.set_ylabel('Error Rate (m/s²)', fontsize=11)
        ax6.set_title('Phase Plot', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance Metrics Table
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        metrics_text = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Mean Absolute Error (MAE)', f'{metrics["mae"]:.4f} m/s', 
             'Average Velocity', f'{metrics["avg_velocity"]:.4f} m/s'],
            ['Root Mean Square Error (RMSE)', f'{metrics["rmse"]:.4f} m/s', 
             'Maximum Error', f'{metrics["max_error"]:.4f} m/s'],
            ['Steady-State Error', f'{metrics["steady_state_error"]:.4f} m/s', 
             'Overshoot', f'{metrics["overshoot"]:.2f}%'],
            ['Settling Time', f'{metrics["settling_time"]:.3f}s' if metrics["settling_time"] else 'N/A', 
             'Response Time', f'{metrics["response_time"]:.3f}s' if metrics["response_time"] else 'N/A'],
        ]
        
        table = ax7.table(cellText=metrics_text, cellLoc='center', loc='center',
                         bbox=[0.1, 0.2, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_text)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        ax7.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Add footer with generation info
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'Generated: {timestamp} | Data points: {len(time)}', 
                ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive analysis saved to {output_file}")
        
        # Also save individual state analysis
        self._plot_state_analysis()
        
        plt.close()
    
    def _add_state_regions(self, ax, time, state):
        """Add colored regions for different states"""
        state_colors = {1: '#90EE90', 2: '#FFD700', 3: '#FFB6C6'}  # Green, Yellow, Red-ish
        state_names = {1: 'MOVING', 2: 'SLOWING', 3: 'STOPPED'}
        
        current_state = state[0]
        start_time = time[0]
        
        for i in range(1, len(state)):
            if state[i] != current_state or i == len(state) - 1:
                end_time = time[i] if i < len(state) - 1 else time[-1]
                ax.axvspan(start_time, end_time, alpha=0.2, 
                          color=state_colors.get(current_state, 'gray'))
                
                current_state = state[i]
                start_time = time[i]
    
    def _plot_state_analysis(self):
        """Create detailed state-by-state analysis"""
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('State-by-State Performance Analysis', fontsize=16, fontweight='bold')
        
        time = np.array(self.data['time'])
        velocity = np.array(self.data['velocity'])
        setpoint = np.array(self.data['setpoint'])
        error = np.array(self.data['error'])
        state = np.array(self.data['state'])
        
        state_names = {1: 'MOVING (GREEN)', 2: 'SLOWING (YELLOW)', 3: 'STOPPED (RED)'}
        state_colors = {1: 'green', 2: 'orange', 3: 'red'}
        
        for idx, (state_val, state_name) in enumerate(state_names.items()):
            ax = axes[idx]
            
            # Filter data for this state
            mask = state == state_val
            if not np.any(mask):
                ax.text(0.5, 0.5, f'No data for {state_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(state_name, fontsize=13, fontweight='bold')
                continue
            
            state_time = time[mask]
            state_velocity = velocity[mask]
            state_setpoint = setpoint[mask]
            state_error = error[mask]
            
            # Normalize time to start at 0 for each state
            state_time = state_time - state_time[0]
            
            # Plot
            ax.plot(state_time, state_setpoint, 'k--', linewidth=2, 
                   label='Target', alpha=0.7)
            ax.plot(state_time, state_velocity, color=state_colors[state_val], 
                   linewidth=1.5, label='Actual', alpha=0.9)
            ax.fill_between(state_time, state_velocity, state_setpoint, 
                           alpha=0.2, color=state_colors[state_val])
            
            # Calculate metrics for this state
            state_mae = np.mean(np.abs(state_error))
            state_rmse = np.sqrt(np.mean(state_error**2))
            
            ax.set_xlabel('Time in State (s)', fontsize=11)
            ax.set_ylabel('Velocity (m/s)', fontsize=11)
            ax.set_title(f'{state_name} - MAE: {state_mae:.4f} m/s, RMSE: {state_rmse:.4f} m/s', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pid_state_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ State analysis saved to pid_state_analysis.png")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\n" + "="*70)
        print("  PID PERFORMANCE ANALYSIS")
        print("="*70)
        
        metrics = self.calculate_metrics()
        
        print("\n📊 Performance Metrics:")
        print(f"  • Mean Absolute Error: {metrics['mae']:.4f} m/s")
        print(f"  • Root Mean Square Error: {metrics['rmse']:.4f} m/s")
        print(f"  • Maximum Error: {metrics['max_error']:.4f} m/s")
        print(f"  • Steady-State Error: {metrics['steady_state_error']:.4f} m/s")
        print(f"  • Average Velocity: {metrics['avg_velocity']:.4f} m/s")
        
        if metrics['settling_time']:
            print(f"  • Average Settling Time: {metrics['settling_time']:.3f}s")
        if metrics['response_time']:
            print(f"  • Average Response Time: {metrics['response_time']:.3f}s")
        
        print(f"  • Maximum Overshoot: {metrics['overshoot']:.2f}%")
        
        print("\n📈 Generating plots...")
        self.plot_comprehensive_analysis()
        
        print("\n" + "="*70)
        print("  Analysis Complete!")
        print("="*70 + "\n")


if __name__ == '__main__':
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'pid_performance.json'
    
    analyzer = PIDAnalyzer(data_file)
    analyzer.generate_all_plots()