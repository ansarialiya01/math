import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# ---------- Part 1: SVD of matrix A ----------
A = np.array([[3, 1, 1], [-1, 3, 1]])
ATA = A.T @ A
print("Original matrix=\n", A)
print("Matrix A^tA=\n", ATA)

# Eigenvalues/eigenvectors of A^T A, sorted in decreasing order
vals, vecs = np.linalg.eig(ATA)
idx = vals.argsort()[::-1]
vals, v = vals[idx], vecs[:, idx]
print("Eigenvalues in decreasing order=\n", np.round(vals, 4))
print("Eigenvectors in decreasing order=\n", np.round(v, 4))

# Sigma
s = np.sqrt(np.clip(vals, 0, None))
m, n = A.shape
sigma = np.zeros((m, n))
np.fill_diagonal(sigma, s[:m])
print("Singular values=\n", np.round(s, 4))
print("Matrix Sigma=\n", np.round(sigma, 4))

# U from A v_i / sigma_i
u = A @ v[:, :m] / s[:m]
print("Matrix U=\n", np.round(u, 4))
print("Matrix V=\n", np.round(v, 4))
print("SVD of matrix A=\n", u @ sigma @ v.T)

# ---------- Part 2: Image compression ----------
image = color.rgb2gray(io.imread("image.webp"))
print(image.shape)

U, S, VT = np.linalg.svd(image)

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray'); plt.title("Original image"); plt.axis("off")

for i, k in enumerate([10, 50, 100], start=2):
    plt.subplot(1, 4, i)
    plt.imshow((U[:, :k] * S[:k]) @ VT[:k, :], cmap='gray')
    plt.title(f"rank {k}"); plt.axis("off")

plt.tight_layout()
plt.show()