# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import jacobi as jac
import scipy.sparse as sp


def main():
    """Using jacobi's algorithm to estimate the eigenvalues
    of the Hamiltonian for a one-electron system with a HO-potential."""
    N = int(eval(input("Number of grid points: ")))
    rho_max = 12.5
    rho = np.linspace(0, rho_max, N)  # step-variable.

    h = (rho[-1]-rho[0])/N  # step size.
    e = -1/h**2  # off-diagonal elements.
    d = 2/h**2 + rho[1:-1]**2  # diagonal elements.
    # creating matrix:
    A = np.diag(d) + sp.diags([e, e], [-1, 1], [N-2, N-2]).toarray()

    mask = np.ones(A.shape, dtype=bool)  # creating mask to conceal diagonal.
    np.fill_diagonal(mask, 0)

    a = np.linalg.norm(A[mask])  # norm of non-diagonal elements.
    # initial indices for rotational elements:
    k = N-1
    l = N-2

    A = jac.jacobi(A, mask, a, k, l, N-2)  # diagonalizing matrix.
    print(np.sort(np.diag(A))[:4])  # printing first 4 eigenvalues.
    # print(np.sort(np.linalg.eig(A)[0])[:4])


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 oneelectron.py
Number of grid points: 100
[ 3.02531219  7.04570702 11.04999135 15.03806645]
"""
