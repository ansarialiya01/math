import numpy as np
import matplotlib.pyplot as plt

# Step 1: data
data = np.array([[1,2],[1,3],[2,2],[2,3],[2,4],[2,5],[3,3],[3,4],[4,4],[4,5]])
print("original data point\n", data)

# Step 2: centering
centered = data - data.mean(axis=0)
print("centered data points:\n", centered)

# Step 3: covariance matrix
cov = centered.T @ centered / (len(centered) - 1)
print("covariance matrix\n", cov)

# Step 4: eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(cov)
print("eigen values:\n", vals)
print("eigen vectors:\n", vecs)

# Step 5: sort in descending order
idx = np.argsort(vals)[::-1]
vals, pcs = vals[idx], vecs[:, idx]
print("sort eigenvalue:\n", pcs)

# Step 6: project on PC1
pc1, pc2 = pcs[:, 0], pcs[:, 1]
data_id = centered @ pc1
print("transformed data:\n", data_id)

# Step 7: PC1 axis and projected points
t = np.linspace(-3, 3, 10)
x1, y1 = t * pc1[0], t * pc1[1]
proj = np.outer(data_id, pc1)
print(proj)

# Step 8: plot
plt.scatter(*centered.T, label="data points")
plt.quiver(0, 0, *pc1, angles='xy', scale_units='xy', scale=1, color='red', label='pc1')
plt.quiver(0, 0, *pc2, angles='xy', scale_units='xy', scale=1, color='green', label='pc2')
plt.plot(x1, y1, label='pc1 - axis')
plt.scatter(*proj.T, c='purple', s=50, label='projected points')
plt.axhline(0, color='black'); plt.axvline(0, color='black')
plt.legend()
plt.show()