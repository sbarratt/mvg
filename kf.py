from mvg import MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt

# x_{t+1} = Ax_t + w_t
# y_{t} = Cx_t + v_t

n = 5
p = 4
Sigma_w = 1*np.eye(n)
Sigma_v = 1*np.eye(p)
A = .96*np.eye(n)+.01*np.random.randn(n,n)
C = np.random.randn(p,n)

xhat = MultivariateGaussian(np.zeros(n),np.eye(n))
w = MultivariateGaussian(np.zeros(n),Sigma_w)
v = MultivariateGaussian(np.zeros(p),Sigma_v)

x = np.random.randn(n)
Xs, Xhats = [], []
for i in range(1000):
    y = C@x + v.sample(1).flatten()
    xy = np.r_[A,C]@xhat + w.concat(v)
    xhat = xy.condition(np.arange(n,p+n),y)
    Xhats.append(xhat.mean)
    Xs.append(x)
    x = A@x + w.sample(1).flatten()
Xhats = np.array(Xhats)
Xs = np.array(Xs)

fig, axes = plt.subplots(5,1)

for i in range(n):
    axes[i].plot(Xhats[:,i], label='xhat')
    axes[i].plot(Xs[:,i], label='x')

plt.legend()
plt.show()