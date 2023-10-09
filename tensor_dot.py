
def Z_dot(lambda1, lambda2, lambda3, G1, G2, U):
    x1 = np.tensordot(lambda1, G1, axes=(1, 1))
    x2 = np.tensordot(lambda2, G2, axes=(1, 1))
    x3 = np.tensordot(x2, lambda3, axes=(2, 0))
    x4 = np.tensordot(x1, x3, axes=(2, 0))
    x5 = np.tensordot(x4, U, axes=([1, 2], [2, 3]))
    return x5

def Z_ein(lambda1, lambda2, lambda3, G1, G2, U):
    Z = np.einsum("ab,de,gh,cbd,feg,ijcf->ahij", lambda1, lambda2, lambda3, G1, G2, U, optimize='greedy')
    return Z
