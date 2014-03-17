import numpy as np

d = 2
n = 100000
D = 1.0
dt = 1.0

x = np.zeros([n, d])
t = 1
while True:
    x += np.random.normal(size=(n, d), scale=np.sqrt(2 * D * dt))
    print(t, np.mean(np.sum(np.square(x), axis=1)) / (2 * d * float(t)))
    t += dt
