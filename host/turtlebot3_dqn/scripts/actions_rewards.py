import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = math.pi

# Create heading using np.linspace
heading = np.linspace(-pi, pi, 100)

# Initialize list to store tr values
tr_values = []

# Iterate over the heading values
for i in range(5):
    angle = -pi/4 + heading + i * pi/8 + pi/2
    
    # Calculate tr value
    tr = 1 - 4 * np.abs(0.5 - np.modf(0.25 + 0.5 * angle % (2*pi) / pi)[0])
    tr_values.append(tr)

# Plotting
for i in range(5):
    plt.plot(heading, tr_values[i], label=f'reward[{i}]')

plt.xlabel('Heading (radian)')
plt.ylabel('rewards')
plt.title('Plot of rewards vs Heading')
plt.legend()
plt.grid(True)
plt.show()
