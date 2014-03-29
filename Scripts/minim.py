#! /usr/bin/env python

from __future__ import print_function, division
import argparse
import numpy as np
import utils
import os
import scipy.spatial
import hcp


# Defaults, best physical estimates
v_propuls_0 = 5.0
Rc = 3.5
Rp = 1.0
visc = 8.9e-10
l_deb = 0.01
alpha = 5.0
T = 10.0
electro_energy = 1e-4

gap = 0.1
c_sep = 2.0 * (Rp + gap)

d0 = gap


def R_rot(th):
    sx, sy, sz = np.sin(th)
    cx, cy, cz = np.cos(th)
    R = np.zeros((3, 3))
    R[0, :] = cy * cz, -cy * sz, sy
    R[1, :] = sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy
    R[2, :] = -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy
    return R


def rotate(u, th):
    return np.sum(u * R_rot(th), axis=-1)


def rot_diff(u, D, dt):
    return rotate(u, np.sqrt(3.0 * D * dt) * np.random.normal(size=3))


def drag_sphere(visc, R):
    return 6.0 * np.pi * visc * R


def D_sphere(T, visc, R):
    return scipy.constants.k * T / drag_sphere(visc, R)


def drag_rot_sphere(visc, R):
    return 8.0 * np.pi * visc * R ** 3


def D_rot_sphere(T, visc, R):
    return scipy.constants.k * T / drag_rot_sphere(visc, R)


def F_to_v(visc, R):
    return 1.0 / (6.0 * np.pi * visc * Rp)


def points_points_sep(rs1, rs2, L):
    # abssep = np.abs(rs1[:, np.newaxis] - rs2[np.newaxis, :])
    # abssep = np.minimum(abssep, L - abssep)
    # sep_sq = utils.vector_mag_sq(abssep)
    sep_sq = scipy.spatial.distance.cdist(rs1, rs2, metric='sqeuclidean')
    return rs1 - rs2[np.argmin(sep_sq, axis=1)]


def angle_wrap(th):
    return (th + np.pi) % (2.0 * np.pi) - np.pi


def wrap(r, L):
        if np.isfinite(L):
            return (r + L / 2.0) % L - L / 2.0
        elif L == np.inf:
            return r


def minim(t_max, dt, L, v_propuls_0=v_propuls_0,
          Rc=Rc, Rp=Rp, visc=visc, l_deb=l_deb, alpha=alpha, T=T,
          electro_energy=electro_energy,
          out=None, every=None, seed=None):
    np.random.seed(seed)
    dim = 3

    r_hcp = hcp.hex_lattice(2) * (2.0 * Rc + c_sep)
    rcs = np.zeros([len(r_hcp), dim])
    rcs[:, :2] = r_hcp

    # Translational diffusion constant
    D = D_sphere(T, visc, Rp)
    l_diff_0 = np.sqrt(2.0 * D * dt)
    v_diff_0 = l_diff_0 / dt

    # Rotational diffusion constant
    D_rot = D_rot_sphere(T, visc, Rp)

    # Hydrodynamic prefactor
    v_hydro_0 = (3.0 / 16.0) * alpha * Rp ** 2 * v_propuls_0

    # Reduced mass
    R_red = (Rp * Rc) / (Rp + Rc)
    # Force prefactor
    Z = electro_energy / R_red
    # Electrostatic prefactor
    F_electro_0 = Z * R_red / l_deb
    # Factor to convert force into speed
    F_to_v_p = F_to_v(visc, Rp)

    if out is not None:
        utils.makedirs_safe(os.path.join(args.out, 'dyn'))
        np.savez(os.path.join(args.out, 'static.npz'),
                 rcs=rcs, Rc=Rc, Rp=Rp, L=L)

    rp = np.array([-(Rp + Rc) - d0] + (dim - 1) * [0.0])
    # up = utils.sphere_pick(n=1, d=dim)[0]
    # start active colloid off perpendicular to fixed colloid
    # up = utils.vector_unit_nonull(np.array([1.0] + (dim - 1) * [-0.01]))
    # start active colloid set to graze fixed colloid tangentially
    theta0 = np.arcsin((Rc + Rp) / (Rc + Rp + d0))
    up = utils.vector_unit_nonull(
        np.array([np.cos(theta0), np.sin(theta0), 0.0]))

    i_t, t = 0, 0.0
    while t < t_max:
        # Propulsion
        v_propuls = up * v_propuls_0

        # Translational diffusion
        v_diff = v_diff_0 * np.random.normal(size=dim)

        v_hydro = v_electro = 0.0
        for rc in rcs:
            # Calculate useful distances
            r_pc = rp - rc
            u_pc = utils.vector_unit_nonull(r_pc)
            s_pc = utils.vector_mag(r_pc)
            h_pc = s_pc - Rc
            d_pc = s_pc - Rc - Rp

            # Electrostatic force (repulsive)
            v_electro += F_to_v_p * F_electro_0 * np.exp(-d_pc / l_deb) * u_pc

            # Hydrodynamics
            # Hydrodynamic torque (aligns parallel)
            # Definitions
            u_o = up
            u_perp = -u_pc
            # Calculations
            mag_u_o_perp = np.dot(u_o, u_perp)
            u_o_perp = mag_u_o_perp * u_perp
            u_o_par = u_o - u_o_perp
            u_par = utils.vector_unit_nonull(u_o_par)
            th_o_perp = np.arccos(mag_u_o_perp)
            omega = v_hydro_0 * -2.0 * np.sin(-2.0 * th_o_perp) / h_pc ** 3
            phi = omega * dt
            th_n_perp = th_o_perp + phi
            mag_u_n_perp = np.cos(th_n_perp)
            mag_u_n_par = np.sqrt(1.0 - mag_u_n_perp ** 2)
            u_n = mag_u_n_perp * u_perp + mag_u_n_par * u_par
            up = u_n.copy()
            # Hydrodynamic force (complicated)
            v_hydro_perp = -2.0 * \
                (1.0 - 3.0 * np.cos(th_o_perp) ** 2) * -u_perp
            v_hydro_par = np.sin(-2.0 * th_o_perp) * u_par
            v_hydro += v_hydro_0 * (v_hydro_par + v_hydro_perp) / h_pc ** 2

        v = v_propuls + v_diff + v_hydro + v_electro

        rp += v * dt
        rp = wrap(rp, L)

        up = rot_diff(up, D_rot, dt)

        if out is not None and not i_t % every:
            np.savez(os.path.join(out, 'dyn/{:010d}'.format(i_t)),
                     t=t, rp=rp, up=up, vh=v_hydro, ve=v_electro, vp=v_propuls)
            print(t)

        i_t += 1
        t += dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate active particles in an HCP lattice')
    parser.add_argument('-o', '--out',
                        help='Output directory')
    parser.add_argument('-t', type=float, default=np.inf)
    parser.add_argument('-dt', type=float)
    parser.add_argument('-e', '--every', type=int)
    parser.add_argument('-L', type=float, default=np.inf)
    args = parser.parse_args()

    # minim(args.t, args.dt, args.L, out=args.out,
    #       every=args.every, seed=1, alpha=0.0)

    seeds = [12]
    alphas = np.linspace(0.0, 8.0, 99)
    header = '\n'.join(
        ['T {:f}'.format(T), 'dt {:f}'.format(args.dt), 't_max {:f}'.format(args.t)])
    for seed in seeds:
        taus = []
        print('seed {}'.format(seed))
        for alpha in alphas:
            tau = minim(args.t, args.dt, args.L,
                        out=args.out, every=args.every, alpha=alpha, T=T, seed=seed)
            print('alpha {} tau {}'.format(alpha, tau))
            taus.append(tau)
            if tau == np.inf:
                break
    # If last leaving time was infinite, assume rest are too
    # but nan instead of inf to show hasn't actually been calculated
        taus += [np.nan] * (len(alphas) - len(taus))
        np.savetxt('../Data/tau_alpha_T/T_{:.2f}_{:d}_temp.csv'.format(
            T, seed), zip(alphas, taus), header=header)
