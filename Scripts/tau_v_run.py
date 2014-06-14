import numpy as np
import multiprocessing
import minim


def tau_of_v(params):
    dt, t, seed = params
    vs = np.linspace(3.0, 20.0, 100)
    taus = []
    for v in vs:
        tau = minim.minim(dt, t, v_propuls_0=v, seed=seed)
        print('seed {} v {} tau {}'.format(seed, v, tau))
        taus.append(tau)
        if tau == np.inf:
            break
    # If last leaving time was infinite, assume rest are too
    # but nan instead of inf to show hasn't actually been calculated
    taus += [np.nan] * (len(vs) - len(taus))
    header = '\n'.join(['dt {:f}'.format(dt), 't_max {:f}'.format(t)])
    np.savetxt('../Data/2d/tau_v_300K/{:d}.csv'.format(
        seed), list(zip(vs, taus)), header=header)


if __name__ == '__main__':
    data = [(0.0001, 100.0, seed) for seed in range(40, 51)]
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus - 1)
    p.map(tau_of_v, data)
