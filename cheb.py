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