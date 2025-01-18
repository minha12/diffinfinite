import re
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

def parse_log_file(log_file):
    """Parses the log file to extract loss values and iteration numbers."""
    losses = []
    iterations = []
    loss_pattern = re.compile(r"loss: (\d+\.\d+):.*\| (\d+)/")

    with open(log_file, 'r') as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                loss = float(match.group(1))
                iteration = int(match.group(2))
                losses.append(loss)
                iterations.append(iteration)
    return losses, iterations

def moving_average(data, window_size):
    """Calculates the moving average of a list."""
    if len(data) < window_size:
        return data  # Return original data if window is larger than data
    
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='valid')
    return smoothed_data

def visualize_loss(losses, iterations, output_path="docs/loss_plot.png", window_size=100):
    """Visualizes the loss values over iterations with a moving average."""
    plt.figure(figsize=(12, 7))

    # Plot raw loss values
    plt.plot(iterations, losses, marker='.', linestyle='-', markersize=2, label='Raw Loss', alpha=0.5)

    # Calculate and plot moving average
    smoothed_losses = moving_average(losses, window_size)
    smoothed_iterations = iterations[window_size - 1:] # Adjust iterations for the moving average
    plt.plot(smoothed_iterations, smoothed_losses, linestyle='-', linewidth=2, label=f'Moving Avg (Window={window_size})', color='red')

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Training Iterations with Moving Average")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.show()

def main(log_file: str, output_path: str = "docs/loss_plot.png", window_size: int = 100):
    losses, iterations = parse_log_file(log_file)
    visualize_loss(losses, iterations, output_path=output_path, window_size=window_size)

if __name__ == "__main__":
    Fire(main)
    # log_file = "logs/srun_20250118_081031.log"  # Replace with your log file name
    # losses, iterations = parse_log_file(log_file)
    # visualize_loss(losses, iterations, window_size=100)