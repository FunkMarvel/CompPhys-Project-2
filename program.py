# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import scipy.sparse as sp

# initializing values:
N = int(eval(input("Number of grid points: ")))
rho = np.linspace(0, 1, N)  # step-variable.

h = (rho[-1]-rho[0])/N  # step size.
a = -1/h**2  # off-diagonal elements.
d = 2/h**2  # diagonal elements.


def main():
    """This program finds the eigenvalues of a NxN Toeplitz matrix,
    and compares them with the analytical values."""
    A = tri_matrix()  # creating matrix.
    eigvals = np.sort(np_eigvals(A)[0])  # retrieving numerical eigenvalues.
    an_egivals = np.sort(anal_eig())  # retrieving analytical eigenvalues.
    test_eigvals(eigvals, an_egivals)  # testing eigenvalue correspondance.
    print(eigvals)
    print(an_egivals)


def tri_matrix():
    """Function that constructs the Toeplitz matrix."""
    A = sp.diags([a, d, a], [-1, 0, 1], [N, N]).toarray()
    return A


def np_eigvals(A):
    """Function that finds eigenvalues and eigenvectors of input matrix,
    using NumPy's built in solver."""
    eigvals = np.linalg.eig(A)
    return eigvals


def anal_eig():
    """Function that returns the analytical eigenvalues of
    a NxN Toeplitz matrix."""
    j_val = np.arange(1, N+1)
    an_eigvals = d - 2*a*np.cos(j_val*np.pi/(N+1))
    return an_eigvals


def test_eigvals(num, an):
    """Testing if NumPy eigenvalues matches analytical eigenvalues
    within the set tolerance."""
    tol = 1e-10
    err = np.max(np.abs(num-an))
    msg = "Analytical and NumPy eigvals don't match, error = %e" % err
    assert tol > err, msg


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 program.py
Number of grid points: 10
"""
# No assertion error. The test passes.
