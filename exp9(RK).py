import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: DE y' = x + y, y(0) = 1
f = lambda x, y: x + y
x0, y0, xn, h = 0, 1, 1, 0.1
n = int((xn - x0) / h)
x = np.linspace(x0, xn, n + 1)

# Step 2 & 3: Euler and RK4
y_euler = np.zeros(n + 1); y_euler[0] = y0
y_rk4 = np.zeros(n + 1);   y_rk4[0] = y0

for i in range(n):
    y_euler[i+1] = y_euler[i] + h * f(x[i], y_euler[i])

    k1 = h * f(x[i], y_rk4[i])
    k2 = h * f(x[i] + h/2, y_rk4[i] + k1/2)
    k3 = h * f(x[i] + h/2, y_rk4[i] + k2/2)
    k4 = h * f(x[i] + h, y_rk4[i] + k3)
    y_rk4[i+1] = y_rk4[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

print(f"Euler: value of y at x={xn} is y={y_euler[-1]}")
print(f"RK4:   value of y at x={xn} is y={y_rk4[-1]}")

# Step 4: exact solution and table
y_actual = 2*np.exp(x) - x - 1
print(pd.DataFrame({'x': x, 'Euler': y_euler, 'RK4': y_rk4, 'Actual': y_actual}))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_euler, 'r--o', label='Euler')
plt.plot(x, y_rk4, 'g--^', label='RK4')
plt.plot(x, y_actual, 'y-s', label='Actual')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
plt.show()