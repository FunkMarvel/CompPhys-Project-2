# Project 2 FYS3150, Anders P. Åsbø
import numpy as np
import program as pro
import jacobi as jac


def main():
    test_find_max_diagonal()

    global N
    N = int(eval(input("Number of grid points: ")))
    h = 1/N
    d = 2/h**2
    e = -1/h**2
    A = pro.tri_matrix(d, e, N)  # creating matrix.
    mask1 = np.ones(A.shape, dtype=bool)  # creating mask to conceal diagonal.
    np.fill_diagonal(mask1, 0)

    a = np.linalg.norm(A[mask1])  # norm of non-diagonal elements.
    # initial indices for rotational elements:
    k = N-1
    l = N-2

    A = jac.jacobi(A, mask1, a, k, l, N)  # solving numerical eigenvalues.
    eigvals = np.sort(np.diag(A))  # retrieving numerical eigenvalues.
    an_egivals = np.sort(pro.anal_eig(d, e, N))  # retrieving analytical eigenvalues.
    test_eigvals(eigvals, an_egivals)  # testing eigenvalue correspondance.


def test_eigvals(num, an):
    """Testing if NumPy eigenvalues matches analytical eigenvalues
    within the set tolerance."""
    tol = 1e-10
    err = np.max(np.abs(num-an))
    msg = "Analytical and NumPy eigvals don't match, error = %e" % err
    assert tol > err, msg


def test_find_max_diagonal():
    """Testing if 'find_max' successfully locates non-diagonal element
    with largest absolute value."""
    A = np.random.random((5, 5))  # generating random matrix.
    np.fill_diagonal(A, 10)  # filling diagonal with largest value.
    # choosing random indices:
    index1, index2 = np.random.choice(5, 2, replace=False)
    A[index1, index2] = -1-np.random.random()  # setting max non-diag element.

    max_value, k, l = jac.find_max(A, 0, 5)  # finding max non-diag element.
    # testing if all returned values match:
    test = (max_value == A[index1, index2]) and (index1 == k and index2 == l)
    msg = "Function 'find_max' did not find maximum non-diagonal element."
    assert test, msg


if __name__ == '__main__':
    main()

# example run:
"""
$ python3 unittests.py
Number of grid points: 50
"""
# no assertions raised, tests passed.
