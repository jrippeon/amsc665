import numpy as np
import cheb

from scipy.interpolate import BarycentricInterpolator

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


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

test_N_values= np.arange(3, 50, 1)
errors= np.zeros_like(test_N_values, dtype=float)
f= lambda x : np.cos(np.pi * x) + 1
g= lambda x: (np.pi**4 + 4 * np.pi**2 + 3) * np.cos(np.pi * x) + 3

for i, N in enumerate(test_N_values):
    x= cheb.get_nodes(N)
    u= cheb.solve_4th_order_homogeneous((3, 0, -4, 0, 1), g, N)
    errors[i]= poly_error(x, u, f)
    plt.plot(x,u)
    plt.plot(x,f(x))
    plt.show()

plt.semilogy(test_N_values, errors)
plt.xlabel('N')
plt.ylabel(r'$\sup_{j} \left|u(x_j) - u_{exact}(x_j)\right|$')
plt.show()