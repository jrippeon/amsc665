import numpy as np
from scipy.interpolate import BarycentricInterpolator
import matplotlib.pyplot as plt

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

    # g can be passed either as a lambda or as a numpy array

    if callable(g):
        F= g(x)
    else:
        assert g.shape == (N+1,), f'g has shape {g.shape}, should have ({N+1},)'
        F= g

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

    return x, u

def solve_4th_order(c, g, N, domain=(-1,1), dirichlet=(0,0), neumann=(0,0)):
    """
    Solve the 4th order BVP

    u : [a, b] --> R
    c_4 u'''' + c_3 u''' + c_2 u'' + c_1 u' + c_0 u = g
    u(-1) = dl
    u(1) = dr
    u'(-1) = nl
    u'(1) = nr

    using Chebyshev spectral methods on N nodes,
    where:

    domain = [a,b]
    dirichlet = [dl, dr]
    neumann = [nl, nr]

    returns y, u
    """

    a= domain[0]
    b= domain[1]
    dl= dirichlet[0]
    dr= dirichlet[1]
    nl= neumann[0]
    nr= neumann[1]

    x= get_nodes(N)
    D= get_diff_matrix(N)
    D2= D@D
    D3= D2@D
    D4= D2@D2
    I= np.eye(N+1)


    # this is the derivative of the linear transformation [a,b] --> [-1,1]
    A = 2/(b - a)

    # transform u_orig on [a,b] to u on [-1, 1], u_orig(x) = u((2x - a - b)/(b-a))

    nl_mod= nl/A
    nr_mod= nr/A
    c_mod= c * np.array([1, A, A**2, A**3, A**4])

    G= g((b-a)/2 * x + (a+b)/2)

    # construct a polynomial w which satisfies the BC

    p0= 0.25 * (x - 1)**2 * (x + 2)
    p1= 0.25 * (x + 1)**2 * (2 - x)
    q0= 0.25 * (x - 1)**2 * (x + 1)
    q1= 0.25 * (x + 1)**2 * (x - 1)

    w= p0 * dl + p1 * dr + q0 * nl_mod + q1 * nr_mod

    # now we decompose u = v + w, so Lu = g becomes Lu = g - Lw

    L= c_mod[4] * D4 + c_mod[3]  * D3 + c_mod[2]  * D2 + c_mod[1] * D + c_mod[0] * I


    # solve this new system
    _, v= solve_4th_order_homogeneous(c_mod, G - L @ w, N)

    # add back in the bc-satisfaction term
    u= v + w

    # rescale space
    y= (b-a)/2 * x + (a+b)/2

    return y, u


def poly_error(x, y, f, M=1000):
    '''
    Return the max error of the polynomial interpolating x and y at approximating the function f.
    Samples at M points.
    '''
    p= BarycentricInterpolator(x,y)
    x_dense= np.linspace(np.min(x), np.max(x), M)
    y_dense= p(x_dense)
    y_exact= f(x_dense)

    return np.max(np.abs(y_dense - y_exact))

def plot_poly(x, u, f=None, M=1000, fig=None, ax=None):
    """
    Plot the Chebyshev polynomial with points (x,u) on M points,
    and optionally compare to exact function f
    """
    if fig is None or ax is None:
        fig, ax= plt.subplots()
    p= BarycentricInterpolator(x,u)
    x_dense= np.linspace(np.min(x), np.max(x), M)
    y_dense= p(x_dense)
    ax.plot(x_dense, y_dense)
    if not f is None:
        y_exact= f(x_dense)
        ax.plot(x_dense, y_exact)
    return fig, ax