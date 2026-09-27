import numpy as np

def get_nodes(N):
    '''
    Obtain the Nth order Chebyshev extrema nodes.
    '''
    theta= np.linspace(0, np.pi, N+1)
    x= np.cos(theta)

    return x

def get_diff_matrix(N):
    '''
    Get the (N+1,N+1) Chebyshev differentiation matrix.
    '''

    x= get_nodes(N)

    a= np.zeros(N+1)

    for i in range(len(a)):
        # prod (x_i - x_k) over all k != i
        a[i]= np.prod(x[i] - x[:i]) * np.prod(x[i] - x[i+1:])

    D= np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j]= a[i] / (a[j] * (x[i] - x[j]))
            # else:
            #     D[i,j]= np.sum(1 / (x[j] - x[:j])) + np.sum(1 / (x[j] - x[j+1:]))
    for i in range(N+1):
        # diagonal gets the negative sum of the rest of the entries
        # this was discussed in spectral methods in matlab
        D[i,i]= -np.sum(D[i])
    return D

def solve_4th_order_homogeneous(c, g, N):
    """
    Solve the 4th order BVP with homogeneous BC

    u : [-1, 1] --> R
    c_4 u'''' + c_3 u''' + c_2 u'' + c_1 u' + c_0 u = f
    u(-1) = u'(-1) = u(1) = u'(1) = 0

    using Chebyshev spectral methods on N nodes.
    """

    I= np.eye(N+1)
    x= get_nodes(N)
    X= np.diag(x)
    X2= X**2
    D= get_diff_matrix(N)
    D2= D @ D
    D3= D2 @ D
    D4= D2 @ D2
    F= g(x)

    # we substitute u = (1 - x^2)v, and solve for v with v(-1) = v(1) = 0
    # this enforces the neumann condition automatically

    L0= (I - X2)
    L1= -2 * X + (I - X2) @ D
    L2= -2 * I - 4 * X @ D + (I - X2) @ D2
    L3= -6 * D - 6 * X @ D2 + (I - X2) @ D3
    L4= -12 * D2 -8 * X @ D3 + (I - X2) @ D4

    L= c[0] * L0 + c[1] * L1 + c[2] * L2 + c[3] * L3 + c[4] * L4

    # modify the 0th and Nth entries to enforce the dirichlet condition
    L[0, :]= np.zeros(N+1)
    L[0,0]= 1
    F[0]= 0
    L[N, :]= np.zeros(N+1)
    L[N,N]= 1
    F[N]= 0

    # solve
    v= np.linalg.solve(L, F)

    u= (I - X2) @ v

    return u