import numpy as np
import cheb

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def p5(N):
    """
    Solve the differential equation in p5 with a specified number of nodes

    u'''' - 4 u'' + 3 u = g

    with homogeneous Dirichlet and Neumann BC
    """

    x= cheb.get_nodes(N)
    D= cheb.get_diff_matrix(N)
    D2= D@D
    D3= D2@D
    D4= D2@D2
    I= np.eye(N+1)

    # this is our boundary condition
    g_func= lambda x: (np.pi**4 + 4 * np.pi**2 + 3) * np.cos(np.pi * x) + 3
    g= g_func(x)

    # We set u = (1-x^2)v, and enforce v(-1) = v(1) = 0
    # This solves both BC, and gives us a new equation
    # (1-x^2) v'''' + (-8x)v''' + (-12-4(1-x^2))v'' + 16x v' + (8+3(1-x^2))v = g
    X= np.diag(x)
    X2= X**2
    L= (I - X2) @ D4 -8 * X @ D3 + (-16 * I + 4 * X2) @ D2 + 16 * X @ D + (11 * I - 3 * X2) @ I

    # Strip first and last conditions, use them to enforce the BC
    # now these rows read v[0] == 0 and v[N] == 0
    L[0,:]= np.zeros(N+1)
    L[0, 0]= 1
    L[N,:]= np.zeros(N+1)
    L[N,N]= 1

    g[0]= 0
    g[N]= 0

    # solve for v
    v= np.linalg.solve(L, g)


    # now re-create u
    u= (I - X2) @ v

    return u


# N=100

# x= cheb.get_nodes(N)
# u= p5(N)
# u_exact= np.cos(np.pi * x) + 1
# plt.plot(x, u)
# plt.plot(x, u_exact)
# plt.show()

test_N_values= np.arange(3, 50, 1)
errors= np.zeros_like(test_N_values, dtype=float)

for i, N in enumerate(test_N_values):
    x= cheb.get_nodes(N)
    u= p5(N)
    u_exact= np.cos(np.pi * x) + 1
    e= np.abs(u - u_exact)
    errors[i]= np.max(e)

plt.semilogy(test_N_values, errors)
plt.xlabel('N')
plt.ylabel(r'$\sup_{j} \left|u(x_j) - u_{exact}(x_j)\right|$')
plt.show()