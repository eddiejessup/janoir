#! /usr/bin/env python

from __future__ import print_function, division
import argparse
import numpy as np
import utils
import os
import scipy.spatial
import qrot


d_max = 1.0

# Defaults, best physical estimates
v_propuls_0 = 5.0
Rc = 3.5
Rp = 1.0
visc = 8.9e-10
l_deb = 0.01
alpha = 5.0
kb = 1.38e-11
T = 0.0
electro_energy = 1e-4


def D_sphere(T, visc, R, kb=kb):
    return (kb * T) / (6 * np.pi * visc * R)


def fric_sphere(visc, R):
    return 8.0 * np.pi * visc * R ** 3


def D_rot_sphere(T, visc, R, kb=kb):
    return kb * T / fric_sphere(visc, R)


def R_rot_2d(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]])


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


def wrap(r, L, wrap_info=False):
    rw = (r + L / 2.0) % L - L / 2.0
    if wrap_info:
        return rw, np.asarray(r > L / 2.0, dtype=np.int) - np.asarray(r < -L / 2.0, dtype=np.int)
    else:
        return rw


def minim(t_max, dt, L, v_propuls_0=v_propuls_0,
          Rc=Rc, Rp=Rp, visc=visc, l_deb=l_deb, alpha=alpha, kb=kb, T=T,
          electro_energy=electro_energy,
          out=None, every=None, seed=None):
    np.random.seed(seed)
    dim = 3

    L_half = L / 2.0

    rc = np.zeros([dim])

    rp = np.zeros([dim])
    up = utils.sphere_pick(n=1, d=dim)
    while True:
        rp = np.random.uniform(-L_half, L_half, size=dim)
        if utils.vector_mag_sq(rp - rc) > (Rc + Rp) ** 2:
            break

    # Translational diffusion constant
    D = D_sphere(T, visc, Rp)
    l_diff_0 = np.sqrt(2.0 * D * dt)
    v_diff_0 = l_diff_0 / dt

    # Rotational diffusion constant
    D_rot = D_rot_sphere(T, visc, Rp)
    th_diff_0 = np.sqrt(2.0 * D_rot * dt)

    # Hydrodynamic prefactor
    print(alpha)
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
                 rc=rc, Rc=Rc, Rp=Rp, L=L)

    rp = np.array([-(Rp + Rc) - 2.0] + (dim - 1) * [0.0])
    # up = utils.sphere_pick(n=1, d=dim)[0]
    up = utils.vector_unit_nonull(np.array([1.0] + (dim - 1) * [-0.01]))

    i_t, t = 0, 0.0
    while t < t_max:
        # Propulsion
        v_propuls = up * v_propuls_0

        # Translational diffusion
        v_diff = v_diff_0 * np.random.normal(size=dim)

        # Calculate useful distances
        r_pc = rp - rc
        u_pc = utils.vector_unit_nonull(r_pc)
        s_pc = utils.vector_mag(r_pc)
        h_pc = s_pc - Rc
        d_pc = s_pc - Rc - Rp

        # Electrostatic repulsion
        v_electro = F_to_v_p * F_electro_0 * np.exp(-d_pc / l_deb) * u_pc

        # Hydrodynamics
        # Particle orientation component towards colloid
        u_perp = np.dot(up, u_pc) * u_pc
        u_perp_norm = utils.vector_unit_nonull(u_perp)
        # Particle orientation component tangential to colloid surface
        u_par = up - u_perp
        u_par_norm = utils.vector_unit_nonull(u_par)
        # Angle between particle orientation and surface-parallel direction
        dth = -utils.vector_angle(up, u_par)
        v_hydro_perp = 2.0 * (3.0 * np.sin(dth) ** 2 - 1.0) * u_perp_norm
        v_hydro_par = np.sin(2.0 * dth) * u_par_norm
        v_hydro = v_hydro_0 * (v_hydro_par + v_hydro_perp) / h_pc ** 2

        # v = v_propuls + v_hydro + v_electro + v_diff
        v = v_propuls + v_electro + v_diff

        rp += v * dt
        rp = wrap(rp, L)

        #  Rotational diffusion
        for axis in np.identity(dim):
            up = qrot.rot_a_to_b(up, axis, th_diff_0 * np.random.normal())

        # Hydrodynamic torque
        om_hydro_geom = -2.0 * np.sin(2.0 * dth)
        om_hydro = v_hydro_0 * om_hydro_geom / h_pc ** 3
        up = qrot.rot_a_to_b(up, u_par_norm, om_hydro * dt)

        if out is not None and not i_t % every:
            np.savez(os.path.join(out, 'dyn/{:010d}'.format(i_t)),
                     t=t, rp=rp, up=up, vh=v_hydro, ve=v_electro, vp=v_propuls)
            print(t)

        i_t += 1
        t += dt
    else:
        return np.inf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate active particles in an HCP lattice')
    parser.add_argument('-o', '--out',
                        help='Output directory')
    parser.add_argument('-t', type=float, default=np.inf)
    parser.add_argument('-dt', type=float)
    parser.add_argument('-e', '--every', type=int)
    parser.add_argument('-L', type=float, default=15.0)
    args = parser.parse_args()

    minim(args.t, args.dt, args.L, out=args.out, every=args.every, seed=0, alpha=0.0)

    seeds = range(20)
    alphas = np.linspace(0.0, 5.0, 30)
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
        np.savetxt('dat/tau_alpha_T/T_{:.2f}_{:d}.csv'.format(
            T, seed), zip(alphas, taus), header=header)
