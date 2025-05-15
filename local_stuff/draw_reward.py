import matplotlib.pyplot as plt
import numpy as np

# Initialize dictionaries to store rewards
rewards = {}

# Read and parse the data
with open('reward_history.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if ':' in line:
            name, value = line.split(':')
            name = name.strip()
            try:
                value = float(value.strip())
                if name not in rewards:
                    rewards[name] = []
                rewards[name].append(value)
            except ValueError:
                continue

# Create subplots
n_rewards = len(rewards)
n_cols = 3  # You can adjust number of columns
n_rows = (n_rewards + n_cols - 1) // n_cols

plt.figure(figsize=(15, 4*n_rows))

for idx, (name, values) in enumerate(rewards.items(), 1):
    plt.subplot(n_rows, n_cols, idx)
    plt.plot(values, 'b-', marker='.')
    plt.title(name)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add min, max, mean annotations
    min_val = min(values)
    max_val = max(values)
    mean_val = np.mean(values)
    plt.text(0.02, 0.98, f'min: {min_val:.4f}\nmax: {max_val:.4f}\nmean: {mean_val:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()