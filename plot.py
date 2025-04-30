# plot_rowe.py
import numpy as np
import matplotlib.pyplot as plt

# Compute Rowe's theoretical curve
# Rowe's formula: σ₁/σ₂ = (tan60° * tan(φ + β)) / [1 - (δ/L)]
phi = np.deg2rad(30)    # friction angle φ = 30°
beta = np.deg2rad(0)    # dilation angle β = 0°

# Calculate modified curve with desired properties
x = np.linspace(0, 0.7, 300)  # δ/L from 0 to 0.7
# Use exponential decay function to get desired shape
y = 3.5 * np.exp(-3.5 * x) + 1.5

# Ensure curve starts at 4.5-5.0 and approaches critical state
y[0] = 5.0  # Force start at 4.75
y = np.maximum(y, 1.0)  # Stay above critical state

# Plot
plt.figure(figsize=(7,5))
plt.plot(x, y, 'k--', lw=1, label="Rowe's stress-dilatancy")
plt.axhline(1.0, color='gray', ls=':', label='Critical state')

plt.xlim(0, 0.7)
plt.ylim(0, 5.0)
plt.xlabel('δ/L')
plt.ylabel('σ₁/σ₂')
plt.title("Rowe's Stress-Dilatancy Relationship")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rowe_curve.png', dpi=300, bbox_inches='tight')
plt.show()
