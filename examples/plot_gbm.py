import matplotlib.pyplot as plt

from src.processes.gbm import simulate_gbm

# generate multiple paths
for i in range(5):
    path = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01, seed=i)
    plt.plot(path)

plt.title("Geometric Brownian Motion Paths")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.show()
