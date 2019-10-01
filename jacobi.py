# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import program as pro

tol = 1e-20


def main():
    """Program uses Jacobi's rotation algorithm to diagonalize a tridiagonal,
    Toeplitz matrix."""
    global N
    N = int(eval(input("Number of grid points: ")))
    h = 1/N  # step-size.
    d = 2/h**2  # diagonal elements.
    e = -1/h**2  # off-diagonal elements.
    A = pro.tri_matrix(d, e, N)  # constructing matrix.
    mask = np.ones(A.shape, dtype=bool)  # creating mask to conceal diagonal.
    np.fill_diagonal(mask, 0)

    a = np.linalg.norm(A[mask])  # norm of non-diagonal elements.
    # initial indices for rotational elements:
    k = N-1
    l = N-2

    A = jacobi(A, mask, a, k, l, N)  # solving eigenvalues.
    print(np.sort(np.diag(A)))


def jacobi(A, mask, a, k, l, N):
    """Function that diagonalizes a matrix using
    Jacobi's rotation algorithm."""
    loops = 0  # counting number of loops

    while a > tol:  # looping over algorithm until tolerance is met.
        akl = 0  # finding max off-diagonal element:
        akl, k, l = find_max(A, akl, N)

        # getting matrix elements used in algorithm:
        akk = A[k, k]
        all = A[l, l]

        tau = (all-akk)/(2*akl)  # calculating cot(2*theta).
        if tau > 0:  # calculating tan(theta), such that |theta|<=pi/4.
            t = 1/(tau+np.sqrt(1+tau**2))
        else:
            t = -1/(-tau+np.sqrt(1+tau**2))
        c = 1/np.sqrt(1+t**2)  # calculating cos(theta).
        s = t*c  # calculating sin(theta).

        for i in range(N):  # calculating non-diagonal matrix elements:
            if (i != k) and (i != l):
                aik = A[i, k]
                ail = A[i, l]
                A[i, k] = aik*c - ail*s
                A[k, i] = A[i, k]
                A[i, l] = ail*c + aik*s
                A[l, i] = A[i, l]

        # calculating diagonal and rotational elements:
        A[k, k] = akk*c**2-2*akl*c*s+all*s**2
        A[l, l] = all*c**2+2*akl*c*s+akk*s**2
        A[k, l] = (akk-all)*c*s+akl*(c**2-s**2)
        A[l, k] = -A[k, l]

        a = np.linalg.norm(A[mask])  # calculating norm of non-diagonal elements.
        loops += 1  # counting loop.

    return A  # returning diagonalized matrix.


def find_max(A, akl, N):
    """Function for finding the non-diagonal element with
    largest absolute value."""
    for i in range(N):
        for j in range(N):
            if (i != j) and (abs(A[i, j]) >= abs(akl)):
                akl = A[i, j]
                k, l = i, j
    return akl, k, l  # returning element with indices.


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 jacobi.py
Number of grid points: 3
[[ 1.80000000e+01  0.00000000e+00 -3.29270630e-20]
 [ 0.00000000e+00  3.07279221e+01 -2.67543178e-12]
 [-3.29270630e-20 -2.67543178e-12  5.27207794e+00]]
"""
# the diagonalized matrix is printed.
