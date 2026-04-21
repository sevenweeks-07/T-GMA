import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('fragmentation_log.csv', names=['Time', 'Active', 'Free', 'FragScore'])

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Plot Fragmentation Score
plt.plot(data['Time'], data['FragScore'], color='red', marker='o', linewidth=2, label='Fragmentation Score ($S$)')
plt.fill_between(data['Time'], data['FragScore'], color='red', alpha=0.15)

plt.title('T-GMA Memory Compaction Event', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Steps (Seconds)', fontsize=12)
plt.ylabel('Fragmentation Score (Auto-Scaled)', fontsize=12)

# Dynamic Y-Axis Scaling
max_score = data['FragScore'].max()
if max_score > 0:
    # Scale slightly above the max value so the spike is clearly visible
    plt.ylim(- (max_score * 0.1), max_score * 1.5) 
else:
    # Fallback if the file is completely zeros
    plt.ylim(-0.1, 1.1)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save it
plt.savefig('fragmentation_graph.png', dpi=300)
print("✅ Auto-scaled graph saved as fragmentation_graph.png")