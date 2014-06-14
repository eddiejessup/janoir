
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl
import utils

def normalize(v, tolerance=0.00001):
    mag2 = utils.vector_mag_sq(v)
    # print(v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v /= mag
    return v


def q_mult(q1, q2):
    return np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                     q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                     q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3],
                     q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]])


def q_conjugate(q):
    q_conj = normalize(q)
    q_conj[1:] *= -1.0
    return q_conj


def qv_mult(q1, v1):
    v1 = normalize(v1)
    q2 = np.array([0.0, v1[0], v1[1], v1[2]])
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def axisangle_to_q(v, theta):
    v = normalize(v)
    q = np.array([np.cos(theta / 2.0), v[0], v[1], v[2]])
    q[1:] *= np.sin(theta / 2.0)
    return q


def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = np.acos(w) * 2.0
    return normalize(v), theta


def rot_a_to_b(a, b, theta):
    v = np.cross(a, b)
    q = axisangle_to_q(v, theta)
    return qv_mult(q, a)


if __name__ == '__main__':
    x_axis_unit = np.array([1, 0, 0])
    y_axis_unit = np.array([0, 1, 0])
    z_axis_unit = np.array([0, 0, 1])
    r1 = axisangle_to_q(x_axis_unit, np.pi / 2)
    r2 = axisangle_to_q(y_axis_unit, np.pi / 2)
    r3 = axisangle_to_q(z_axis_unit, np.pi / 2)

    v = qv_mult(r1, y_axis_unit)
    v = qv_mult(r2, v)
    v = qv_mult(r3, v)

    v1 = normalize(np.array([0.1, 0.1, 0.0]))
    n = 200
    for _ in range(n):
        pp.scatter(v1[0], v1[1], c=mpl.cm.jet(int(round(_*(256.0/n)))), s=100)
        v1_old = v1[:]
        v1 = rot_a_to_b(v1_old, y_axis_unit, 0.01)
        print(utils.vector_angle(np.array(v1_old), np.array(v1)))
    pp.xlim(-1.0, 1.0)
    pp.ylim(-1.0, 1.0)
    pp.show()
