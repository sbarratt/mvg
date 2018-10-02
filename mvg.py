import numpy as np
from scipy.linalg import block_diag

class MultivariateGaussian():
    __array_priority__ = 1000000
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
            
    @property
    def n(self):
        return self.mean.shape[0]

    @property
    def shape(self):
        return self.n
        
    @property
    def entropy(self):
        return self.n/2.+self.n/2.*np.log(2*np.pi)+1./2*np.log(np.linalg.det(self.cov))
    
    def pdf(self, x):
        return np.exp(-.5*(x-self.mean)@np.linalg.solve(self.cov, x-self.mean))/np.sqrt((2*np.pi)**self.n*np.linalg.det(self.cov))
    
    def sample(self, N=1):
        if N == 1:
            return np.random.multivariate_normal(self.mean, self.cov)
        else:
            return np.random.multivariate_normal(self.mean, self.cov, size=N)
    
    def condition(self, indices, a):
        x2indices = np.zeros(self.n, np.bool)
        x2indices[indices] = 1
        x1indices = ~x2indices
        
        mu1 = self.mean[x1indices]
        mu2 = self.mean[x2indices]
        Sigma11 = self.cov[x1indices,:][:,x1indices]
        Sigma12 = self.cov[x1indices,:][:,x2indices]
        Sigma21 = self.cov[x2indices,:][:,x1indices]
        Sigma22 = self.cov[x2indices,:][:,x2indices]
        
        mu = mu1+Sigma12@np.linalg.solve(Sigma22,a-mu2)
        Sigma = Sigma11-Sigma12@np.linalg.solve(Sigma22,Sigma21)
        
        return MultivariateGaussian(mu, Sigma)
    
    # add
    def __add__(x, b):
        if isinstance(b, MultivariateGaussian):
            return np.c_[np.eye(x.shape),np.eye(b.shape)]@x.concat(b)
        else:
            return MultivariateGaussian(x.mean+b,x.cov)
    __radd__ = __add__
    
    # rmatmul
    def __rmatmul__(x, A):
        mu = A@x.mean
        Sigma = A@x.cov@A.T
        return MultivariateGaussian(mu, Sigma)
    
    # mul
    def __mul__(x, a):
        if isinstance(a, (int, float)):
            return x.__rmatmul__(a*np.eye(x.n))
        else:
            return NotImplemented
    __rmul__ = __mul__

    # div
    def __div__(x, a):
        return x.__mul__(1./a)
    __truediv__ = __div__
    __rtruediv__ = __truediv__
        
    # sub
    def __sub__(x, b):
        return x.__add__(-b)
    __rsub__ = __sub__
    
    # neg
    def __neg__(x):
        return x.__mul__(-1)
    
    def __getitem__(self, i):
        if isinstance(i, list):
            indices = np.array(i)
            return MultivariateGaussian(self.mean[i], self.cov[i,:][:,i])
        if isinstance(i, np.ndarray):
            return MultivariateGaussian(self.mean[i], self.cov[i,:][:,i])

    def __repr__(self):
        return "mean: %s\ncov: %s" % (self.mean, self.cov)
    
    def __eq__(x, y):
        return np.allclose(x.mean,y.mean) and np.allclose(x.cov,y.cov)
    
    def __copy__(self):
        return MultivariateGaussian(self.mean.copy(), self.cov.copy())
    def __deepcopy__(self):
        return MultivariateGaussian(self.mean.deepcopy(), self.cov.deepcopy())
    
    def __index__(self, i):
        print ("index", i)
        
    def __len__(self):
        return self.n
    
    def concat(self, x):
        # must be independent
        mean = np.r_[self.mean, x.mean]
        cov = block_diag(self.cov, x.cov)
        return MultivariateGaussian(mean, cov)

if __name__ == '__main__':
    #tests
    from scipy.stats import multivariate_normal

    np.random.seed(0)

    for _ in range(2):
        n = 10
        x = MultivariateGaussian(np.zeros(n), np.eye(n))

        x_npy = multivariate_normal(np.zeros(n), np.eye(n))

        assert np.allclose(x.entropy, x_npy.entropy())
        input = np.random.randn(n)
        assert np.allclose(x.pdf(input), x_npy.pdf(input))

        xplus = x+np.ones(n)
        assert np.allclose(xplus.mean, np.ones(n))
        xtimes = .2*xplus
        assert np.allclose(xtimes.mean, .2*np.ones(n))
        assert np.allclose(xtimes.cov, .2*.2*np.eye(n))
        xtimes = xplus/5
        assert np.allclose(xtimes.mean, .2*np.ones(n))
        assert np.allclose(xtimes.cov, .2*.2*np.eye(n))

        assert x.n == n
        assert x.shape == n

        assert x.sample().shape[0] == n
        assert x.sample(100).shape == (100,n)
        assert np.allclose(x.sample(1000000).mean(axis=0), x.mean, rtol=1e-2,atol=1e-2)
        assert np.allclose(np.cov(x.sample(1000000).T), x.cov, rtol=1e-2,atol=1e-2)

        assert np.eye(n)@x == x
        assert (np.ones((1,n))@x).cov == 10