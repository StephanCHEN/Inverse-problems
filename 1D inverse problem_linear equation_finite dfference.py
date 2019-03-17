import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


n = 100 ##number of intervals
alpha = np.ones(n) ##initial values
step = 0.01 ##step size for optimization
max_iter = 10000000
tolerance = 1e-6

#write the values of A, B, C as well as x
B = np.zeros((n, n))
for i in range(1, n):
    B[i, i-1] = 1.0
    
A = -(n+1) * B + n * np.identity(n)
A[0,0] = 1.0

C = np.zeros(n)
C[0] = 1.0

x = np.asarray(range(n))/n
###########################################################################3
def pde_solver(alpha):
        
    return np.linalg.solve(A, np.dot(B, alpha*x)+C)
    

def J(Y, F):
    h = (Y-F)**2
    
    return np.sum(h / n)
    

def J_grad_Y(Y):
    
    return 2 * (Y - F) / n
    
def J_grad_alpha(Y, alpha):
    p = np.linalg.solve(A.T, J_grad_Y(Y))
    
    return np.dot(np.dot(B, np.diag(x)).T, p)

def plot_graph(solution, name):
    plt.subplot(1, 1, 1)
    plt.plot(x, solution, x, F)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().legend(('Solution','Observation'))
    plt.savefig('Solution and observation - {}'.format(name))
###################################################################3
F = (x + 1)*np.exp(x)
Y0 = Y = pde_solver(alpha)


iteration = 0
Err= J(Y, F)
loss = list()
Y_list = list()
Y_list.append(Y)
start_time = time.time()

for k in range(max_iter):
    loss.append(Err)
    iteration += 1
    alpha1 = alpha - step * J_grad_alpha(Y, alpha)
    Y = pde_solver(alpha1)
    Y_list.append(Y)

    err = J(Y, F)
    alpha = alpha1
    if (abs(err-Err)/Err < tolerance):
        break
    Err = err

run_time = time.time() - start_time
loss.append(Err)

plot_graph(Y0, 'original')
plot_graph(Y, 'optimized')

plt.plot(loss)
plt.xlabel('iteration')
plt.ylabel('J')

print('alpha=', alpha)
print('iteration=', iteration)
print('err', err)
print('run time', run_time)