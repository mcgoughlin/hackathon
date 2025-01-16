import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha_initial = 0.01
alpha_max = 0.1
k_warmup = 30
gamma = 1
k_max = 200  # Total number of epochs

# Epochs array
epochs = np.arange(0, k_max + 1)

# Initialize learning rate array
learning_rates = np.zeros_like(epochs, dtype=float)

# Warm-up phase
warmup_phase = epochs <= k_warmup
learning_rates[warmup_phase] = alpha_initial + ((alpha_max - alpha_initial) / k_warmup) * epochs[warmup_phase]

# Decay phase
decay_phase = epochs > k_warmup
learning_rates[decay_phase] = alpha_max * np.exp(-gamma * ((epochs[decay_phase] - k_warmup)/(k_max - k_warmup)))

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(epochs, learning_rates, label='Learning Rate', color='blue', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.ylim(0, alpha_max + 0.01)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_rate_schedule.png')
plt.show()
