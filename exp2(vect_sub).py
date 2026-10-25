import numpy as np
import matplotlib.pyplot as plt

# Plot y = log(x)
x = np.arange(1, 7)
plt.figure(figsize=(3, 3))
plt.plot(x, np.log(x), 'ro--', label='y log(x)')
plt.xlabel('X-axis'); plt.ylabel('Y-axis'); plt.title('Plotting y=log(x)')
plt.legend(); plt.show()

# Feature space: 2D grid, colored by quadrant
g = np.linspace(-1, 1, 10)
u, v = np.meshgrid(g, g)
P = np.vstack([u.ravel(), v.ravel()])
colors = np.select([(P[0] >= 0) & (P[1] >= 0), (P[0] < 0) & (P[1] >= 0), (P[0] < 0) & (P[1] < 0)],
                   ['red', 'green', 'blue'], 'yellow')

plt.figure(figsize=(4, 4))
plt.scatter(*P, c=colors); plt.title('Original Feature Space'); plt.show()

# Transformation matrices
t = np.pi / 4
T = {
    'Scaled':    np.array([[2, 0], [0, 0.5]]),
    'Rotated':   np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]),
    'Reflected': np.array([[1, 0], [0, -1]]),
    'Projected': np.array([[1, 0], [-1, 0]]),
}

# Apply and plot all four
plt.figure(figsize=(8, 8))
for i, (name, M) in enumerate(T.items(), 1):
    Q = M @ P
    plt.subplot(2, 2, i)
    plt.scatter(*Q, c=colors)
    plt.title(f'{name} Feature Space')
plt.show()