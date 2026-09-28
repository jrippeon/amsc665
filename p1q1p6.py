import numpy as np
import cheb
import matplotlib.pyplot as plt

c= np.array([ 1, 1, 0, 0, 1 ])
g= lambda x : (24 - 240*x**2 + 120 * x**4)/(1 + x**2)**5 + 1/(1 + x**2) - 2 * x / (1 + x**2)**2
domain= (0, 5)
dirichlet= (1, 1/26)
neumann= (0, -10/676)

f= lambda x : 1/ (1 + x**2)

# test_N_values= np.arange(3, 50, 5)
test_N_values= np.array([3, 4, 5, 6, 7, 10, 20])
errors= np.zeros_like(test_N_values, dtype=float)

fig, ax= plt.subplots()

for i, N in enumerate(test_N_values):
    x, u= cheb.solve_4th_order(c, g, N, domain=domain, neumann=neumann, dirichlet=dirichlet)
    errors[i]= cheb.poly_error(x, u, f)
    cheb.plot_poly(x, u, fig=fig, ax=ax)

x_dense= np.linspace(domain[0], domain[1], 1000)
ax.plot(x_dense, f(x_dense))
ax.legend([f'Order {N}' for N in test_N_values] + ['Exact'])
ax.set(
    xlabel='x',
    ylabel='u'
)
plt.show()

# plt.semilogy(test_N_values, errors)
# plt.xlabel('N')
# plt.ylabel(r'$\sup_{j} \left|u(x_j) - u_{exact}(x_j)\right|$')
# plt.show()