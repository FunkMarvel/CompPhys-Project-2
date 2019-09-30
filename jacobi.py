# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import program as pro

tol = 1e-10

def main():

    A = pro.tri_matrix()
    N = len(A[0, :])
    mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    a = np.linalg.norm(A[mask])
    k = N-1
    l = N-2
    A = jacobi(A, mask, a, k, l, N)
    print(A)


def jacobi(A, mask, a, k, l, N):
    loops = 1
    akl = A[k, l]

    while a > tol:
        if loops > 1:
            akl = 0
            akl, k, l = find_max(A, akl, N)

        akk = A[k, k]
        all = A[l, l]

        tau = (all-akk)/(2*akl)
        if tau > 0:
            t = 1/(tau+np.sqrt(1+tau**2))
        else:
            t = -1/(-tau+np.sqrt(1+tau**2))

        c = 1/np.sqrt(1+t**2)
        s = t*c

        for i in range(N):
            if (i != k) and (i != l):
                aik = A[i, k]
                ail = A[i, l]
                A[i, k] = aik*c - ail*s
                A[k, i] = aik*c - ail*s
                A[i, l] = ail*c + aik*s
                A[l, i] = ail*c + aik*s

        A[k, k] = akk*c**2-2*akl*c*s+all*s**2
        A[l, l] = all*c**2+2*akl*c*s+akk*s**2
        A[k, l] = 0
        A[l, k] = 0
        a = np.linalg.norm(A[mask])
        loops += 1
    return A


def find_max(A, akl, N):
    for i in range(N):
        for j in range(N):
            if (i != j) and (abs(A[i, j]) >= abs(akl)):
                akl = A[i, j]
                k = i
                l = j
    return akl, k, l


if __name__ == '__main__':
    main()
