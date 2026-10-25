import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2
a, b = 0, 2

plt.figure(figsize=(10, 10))
for i, n in enumerate([4, 10, 50, 100]):
    dx = (b - a) / n
    x_right = np.linspace(a, b, n + 1)[1:]      # right end points
    area = np.sum(f(x_right) * dx)

    plt.subplot(2, 2, i + 1)
    xc = np.linspace(a, b, 50)
    plt.plot(xc, f(xc), label='y=x^2')
    plt.bar(x_right, f(x_right), width=-dx, align='edge', edgecolor='black')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f"No. of rectangles={n}\nArea={area}")
    plt.legend()

plt.tight_layout()
plt.show()