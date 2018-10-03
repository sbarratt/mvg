# mvg: Multivariate Gaussian in numpy

This library serves as a convenient way to manipulate multivariate gaussian vectors in numpy.

### Pdf, Sampling, Entropy

```python
x = MultivariateGaussian(np.array([1,2]), np.eye(2))
x.sample(3) # take 3 samples of x
print (x.entropy) # evaluates to the differential entropy
x.pdf(np.random.randn(2)) # evaluate the pdf at this input
```
### Affine transformations

```python
x = MultivariateGaussian(np.array([1,2]), np.eye(2))
A = np.random.randn(10,2)
b = np.random.randn(10)
y = A@x+b # gives another MultivariateGaussian with the correct mean and covariance
y = 2*x # broadcasts correctly for multiplication
y = x+3 # and addition
y = x-2 # and subtraction
y = x/2 # and division
y = x[0] # and indexing
y = x.concat(x) # and concatenation (assuming independence)
```

### Conditioning
We can condition a multivariate Gaussian vector on realizations of some of its values.
We do this by calling `x.condition(indices, values)`.
For example:
```python
x = MultivariateGaussian(np.array([1,2]), np.eye(2))
y = x.condition([0],np.array([1]))
```

### Application: Kalman Filtering
See `kf.py`.