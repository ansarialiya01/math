import numpy as np
import matplotlib.pyplot as plt

# ---------- Part 1: surface + tangent plane ----------
x = np.linspace(-10, 10, 15)
X, Y = np.meshgrid(x, x)
Z = X**2 + Y**2
Zx, Zy = 2*X, 2*Y

x0 = y0 = 5
z0 = x0**2 + y0**2
Z_tgt = 2*x0*(X - x0) + 2*y0*(Y - y0) + z0

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, label='z=x^2+y^2')
ax.plot_surface(X, Y, Z_tgt, label='tangent plane')
ax.set(xlabel='x', ylabel='y', zlabel='z')
ax.legend()

# ---------- Part 2: level curves + gradient vectors ----------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
cs = plt.contour(X, Y, Z, levels=10)
plt.clabel(cs, inline=True, fontsize=10)
plt.title('Level curve'); plt.xlabel('x'); plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.quiver(X, Y, Zx, Zy)
plt.title('Gradient vectors'); plt.xlabel('x'); plt.ylabel('y')

# ---------- Part 3: level surfaces of f(x,y,z)=x^2+y^2-z ----------
g = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(g, g)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for k in [0, -5, 10]:
    ax.plot_surface(X, Y, X**2 + Y**2 - k, alpha=0.7, label=f'Level curve for k={k}')
ax.set(title='Level surface of f(x,y,z)', xlabel='x', ylabel='y', zlabel='z')
ax.legend()

plt.show()  