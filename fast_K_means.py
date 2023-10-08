

def dist_i(x, mu):
    return np.sum((x-mu)**2, axis=1)


def dist_ij(x, mu):
    dist = np.zeros((x.shape[0], mu.shape[0]))
    for i in range(mu.shape[0]):
        dist[..., i] = np.sum((x-mu[i])**2, axis=1)
    return dist