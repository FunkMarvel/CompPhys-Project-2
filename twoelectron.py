# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import jacobi as jac
import scipy.sparse as sp


def main():
    """Using jacobi's algorithm to estimate the eigenvalues
    of the Hamiltonian for a non-interacting,
    two-electron system with a HO-potential."""
    N = int(eval(input("Number of grid points: ")))
    rho_max = 15.794
    rho = np.linspace(0, rho_max, N)  # step-variable.

    h = (rho[-1]-rho[0])/N  # step size.
    omega_r = int(eval(input("Omega_r: ")))
    e = -1/h**2  # off-diagonal elements.
    d = 2/h**2 + (omega_r**2)*rho[1:-1]**2 + 1/rho[1:-1]  # diagonal elements.
    # creating matrix:
    A = np.diag(d) + sp.diags([e, e], [-1, 1], [N-2, N-2]).toarray()

    mask = np.ones(A.shape, dtype=bool)  # creating mask to conceal diagonal.
    np.fill_diagonal(mask, 0)

    a = np.linalg.norm(A[mask])  # norm of non-diagonal elements.

    A = jac.jacobi(A, mask, a, N-2)  # finding eigenvalues.
    print(np.sort(np.diag(A))[0])  # printing lowest eigenvalue.
    # print(np.sort(np.linalg.eig(A)[0])[:4])


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 twoelectron.py
Number of grid points: 100
Omega_r: 1
4.075455544319714
"""
