import numpy as np
import qrot

d = 3
n = 5000
D = 1.0
dt = 0.00001

u = np.ones([n, d]) * np.sqrt(3.0) / 3.0
u0 = u.copy()
t = dt
while True:
    for axis in np.identity(d):
        for i in range(n):
            u[i] = qrot.rot_a_to_b(u[i], axis, np.random.normal(scale=np.sqrt(2 * D * dt)))
    th = np.arccos(np.sum(u * u0, axis=1))
    print(t, np.mean(np.square(th)) / (2 * d * float(t)))
    t += dt
