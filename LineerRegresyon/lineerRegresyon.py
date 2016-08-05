import numpy as np
import pylab as py


def compute_cost(X, y, theta):
	
	m = y.size
	
	hx = (X.dot(theta).flatten() - y)**2
	J = ( 1.0 / (2*m)) * hx.sum()

	return J

def gradient_descent(X, y, alpha, theta, iter):

	m = y.size

	J_history = np.zeros(shape= (iter, 1))

	for i in range(iter):
		
		hx0 = (X.dot(theta).flatten() - y) 
		hx1 = (X.dot(theta).flatten() - y) * X[:, 1]

		theta[0,0] = theta[0,0] - alpha * (1.0 / m) * hx0.sum()
		theta[1,0] = theta[1,0] - alpha * (1.0 / m) * hx1.sum()

		J_history[i, 0] = compute_cost(X, y, theta)

	return theta, J_history

data_set = np.loadtxt('x01.txt')

X = data_set[:, 0]
y = data_set[:, 1]

m = y.size

XX = np.ones(shape=(m,2))
XX[:, 1] = X

alpha = 0.01
theta = np.zeros(shape = (2,1))
iter = 2000

theta, J =  gradient_descent(XX, y, alpha, theta, iter)

print theta

result = XX.dot(theta).flatten()
py.scatter(X, y, marker= 'x', c='r')
py.plot(X, result)
py.show()
