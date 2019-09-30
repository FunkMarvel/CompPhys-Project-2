# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import program as pro
import jacobi as jac


def main():
    global N
    N = int(eval(input("Number of grid points: ")))
    h = 1/N
    d = 2/h**2
    e = -1/h**2
    A = pro.tri_matrix(d, e, N)  # creating matrix.
    mask1 = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(mask1, 0)
    mask2 = np.zeros(A.shape, dtype=bool)
    np.fill_diagonal(mask2, 1)

    a = np.linalg.norm(A[mask1])
    k = N-1
    l = N-2

    A = jac.jacobi(A, mask1, a, k, l, N)
    eigvals = np.sort(A[mask2])  # retrieving numerical eigenvalues.
    an_egivals = np.sort(pro.anal_eig(d, e, N))  # retrieving analytical eigenvalues.
    test_eigvals(eigvals, an_egivals)  # testing eigenvalue correspondance.


def test_eigvals(num, an):
    """Testing if NumPy eigenvalues matches analytical eigenvalues
    within the set tolerance."""
    tol = 1e-10
    err = np.max(np.abs(num-an))
    msg = "Analytical and NumPy eigvals don't match, error = %e" % err
    assert tol > err, msg


def test_orthogonality():
    pass


if __name__ == '__main__':
    main()
