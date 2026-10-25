import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Part 1: distance functions
L1   = lambda P, Q: np.sum(np.abs(np.subtract(P, Q)))
L2   = lambda P, Q: np.linalg.norm(np.subtract(P, Q))
Lmax = lambda P, Q: np.max(np.abs(np.subtract(P, Q)))
cosine = lambda P, Q: 1 - np.dot(P, Q) / (np.linalg.norm(P) * np.linalg.norm(Q))

P, Q = [1, 3, -1, 5], [-2, 1, -3, 4]
print("L1 distance: ", L1(P, Q))
print("L2 distance: ", L2(P, Q))
print("L max distance: ", Lmax(P, Q))
print("cosine distance: ", cosine(P, Q))

# Part 2: KNN
data = [([1, 2], 0), ([2, 3], 0), ([3, 3], 0), ([6, 7], 1), ([7, 8], 1), ([8, 8], 1)]
query = [4, 5]

distances = [(L2(query, p), p, l) for p, l in data]
for item in distances:
    print(item)

distances.sort(key=lambda t: t[0])
for d, p, l in distances:
    print(f"distances: {np.round(d, 2)}, points:{p}, label: {l}")

k = 3
nearest = distances[:k]
labels = [l for _, _, l in nearest]
print(f"3 nearest neighbours : {nearest}")
print(f" label of 3 nearest neighbours : {labels}")
print(f"Group of query: {Counter(labels).most_common(1)[0][0]}")

# Plot
pts = np.array([p for p, _ in data])
y = np.array([l for _, l in data])
plt.scatter(*pts[y == 0].T, label="Group 0")
plt.scatter(*pts[y == 1].T, label="Group 1")
plt.scatter(*query, label="query")
plt.legend()
plt.show()