# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import jacobi as jac
import scipy.sparse as sp


def main():
    """Using jacobi's algorithm to estimate the eigenvalues
    of the Hamiltonian for a one-electron system with a HO-potential."""
    N = int(eval(input("Number of grid points: ")))
    rho_max = 12.5  # setting max value for position.
    rho = np.linspace(0, rho_max, N)  # step-variable.

    h = (rho[-1]-rho[0])/N  # step size.
    e = -1/h**2  # off-diagonal elements.
    d = 2/h**2 + rho[1:-1]**2  # diagonal elements.
    # creating matrix:
    A = np.diag(d) + sp.diags([e, e], [-1, 1], [N-2, N-2]).toarray()

    mask = np.ones(A.shape, dtype=bool)  # creating mask to conceal diagonal.
    np.fill_diagonal(mask, 0)

    a = np.linalg.norm(A[mask])  # norm of non-diagonal elements.

    A = jac.jacobi(A, mask, a, N-2)  # diagonalizing matrix.
    eigvals = np.sort(np.diag(A))[:4]
    print(eigvals)  # printing first 4 eigenvalues.
    # print(np.sort(np.linalg.eig(A)[0])[:4])
    anal_eigvals = np.array([3, 7, 11, 15])
    error = abs((eigvals-anal_eigvals)/anal_eigvals)  # relative error.
    print(error)


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 oneelectron.py
Number of grid points: 2e2
[ 3.01384183  7.02900535 11.04021305 15.04745905]
[0.00461394 0.00414362 0.00365573 0.00316394]
"""
