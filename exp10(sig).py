import numpy as np
import matplotlib.pyplot as plt

T = 2 * np.pi
t = np.linspace(0, T, 1000)
f = np.sign(np.sin(t))

def fourier_series(N):
    n = np.arange(1, N + 1)[:, None]                  # shape (N,1)
    an = 2/T * np.trapezoid(f * np.cos(n*t), t, axis=1)
    bn = 2/T * np.trapezoid(f * np.sin(n*t), t, axis=1)
    a0 = np.trapezoid(f, t) / T
    return a0/2 + an @ np.cos(n*t) + bn @ np.sin(n*t)

plt.figure(figsize=(10, 6))
plt.plot(t, f, label='Square Wave')
for N in [5, 10, 20, 50]:
    plt.plot(t, fourier_series(N), label=f'Fourier Series (N={N})')
plt.legend(); plt.grid(True)
plt.xlabel('Time'); plt.ylabel('Amplitude')
plt.title('Fourier Series Approximation of Square Wave')
plt.show()