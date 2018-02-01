import numpy as np
import pandas as pd


df = pd.read_csv('~/Ipynb/2/Shrey sir/train.csv', sep=',')

class GradientDescent():
	def __init__(self, alpha=0.000003, tolerance=0.02, max_iterations=2000):
		self._alpha = alpha   #implement learning rate decay. do this later
		self._tolerance = tolerance
		self._max_iterations = max_iterations
		self._thetas = None
		self._bias= 0.1

	def fit(self, xs, ys):
		xs_transposed = xs.transpose()
		num_examples, num_features=np.shape(xs)
		for i in range(self._max_iterations):
			
			dproduct = np.dot(xs,self._thetas)
			diffs=dproduct-ys
			diffs=diffs+self._bias
			
			cost = np.sum(diffs**2) / (2*num_examples)
			#print cost
			gradient = np.dot(xs_transposed, diffs) / num_examples
			gradient_bias = np.sum(diffs)/num_examples
			
			
			self._thetas = self._thetas - self._alpha*gradient
			self._bias = self._bias - self._alpha*gradient_bias
			
			if cost < self._tolerance:
				return self._thetas

		return self._thetas, self._bias

    
	def minibatch_fit(self, xs, ys,minibatchsize):
		num_examples, num_features = np.shape(xs)
		self._thetas = np.random.rand(num_features)
		for i in range(num_examples/minibatchsize):
			xmini=xs[i*minibatchsize:(1+i)*minibatchsize]
			ymini=ys[i*minibatchsize:(1+i)*minibatchsize]
			self._thetas, self._bias = self.fit(xmini,ymini)
		return self._thetas, self._bias

	def predict(self, x):
		return np.dot(x, self._thetas)+self._bias
    
gd = GradientDescent(tolerance=0.022)
xinput = df[df.columns[2:26]]
youtput=df['Output']
thetas,bias=gd.minibatch_fit(xinput,youtput,10000)
print thetas
print bias
ycalculated=gd.predict(xinput)
print ycalculated
