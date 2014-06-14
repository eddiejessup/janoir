import numpy as np
import multiprocessing
import minim


def tau_of_alpha(params):
    dt, t, seed = params
    alphas = np.linspace(0.0, 10.0, 100)
    taus = []
    for alpha in alphas:
        tau = minim.minim(dt, t, alpha=alpha, seed=seed)
        print('seed {} alpha {} tau {}'.format(seed, alpha, tau))
        taus.append(tau)
        if tau == np.inf:
            break
    # If last leaving time was infinite, assume rest are too
    # but nan instead of inf to show hasn't actually been calculated
    taus += [np.nan] * (len(alphas) - len(taus))
    header = '\n'.join(['dt {:f}'.format(dt), 't_max {:f}'.format(t)])
    np.savetxt('../Data/2d/tau_alpha_300K/{:d}.csv'.format(
        seed), list(zip(alphas, taus)), header=header)


if __name__ == '__main__':
    data = [(0.0001, 100.0, seed) for seed in range(29)]
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus - 1)
    p.map(tau_of_alpha, data)
