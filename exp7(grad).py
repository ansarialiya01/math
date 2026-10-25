import numpy as np
import matplotlib.pyplot as plt

# Step 1: function and gradient
f = lambda xy: (xy[0] - 3)**2 + 3*(xy[1] + 1)**2
grad = lambda xy: np.array([2*(xy[0] - 3), 6*(xy[1] + 1)])

# Step 2: gradient descent
def gradient_descent(x0y0, alpha, eps):
    xy = np.array(x0y0, dtype=float)
    path = [xy.copy()]
    i = 1
    while np.linalg.norm(grad(xy)) > eps:
        xy = xy - alpha * grad(xy)
        path.append(xy.copy())
        i += 1
    return np.array(path), i

# Step 3: run from [-5, 5]
pts, i = gradient_descent([-5, 5], alpha=0.1, eps=0.0001)
minima_point, minima_value = pts[-1], f(pts[-1])
print(f"Point of minima: {minima_point}")
print(f"Minima value of the function: {minima_value}")
print(f"Number of iterations: {i}")

# Grid for plots
X, Y = np.meshgrid(np.linspace(-6, 8, 50), np.linspace(-7, 5, 50))
Z = f([X, Y])

# Step 4: 2D plot
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=15)
plt.plot(pts[:, 0], pts[:, 1], 'r-', marker='o')
plt.scatter(*minima_point, color='blue', s=50, label='Minima')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.title("Gradient Descent Algorithm")

# Step 5: 3D plot
plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.contour(X, Y, Z, levels=15)
ax.plot3D(pts[:, 0], pts[:, 1], [f(p) for p in pts], 'r-', marker='o', label='Path')
ax.set(xlabel='x', ylabel='y', zlabel='z', title="Gradient Descent Algorithm")
ax.legend()

plt.show()   # sirf ek baar, end me