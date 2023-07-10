import numpy as np 

# Define the shapes of the tensors
T = 3
M = 4
D = 2

# Create the tensors X and W
X = np.random.randn(T, M, D)
W = np.random.randn(T, M)

# Calculate mu using np.einsum
mu_einsum = np.einsum('tmd,tm->d', X, W)

# Calculate mu using for loop
mu_for_loop = np.zeros(D)
for t in range(T):
    for m in range(M):
        mu_for_loop += W[t, m] * X[t, m, :]

# Print the results
print("mu (np.einsum):")
print(mu_einsum)
print("mu (for loop):")
print(mu_for_loop)


# Calculate cov using np.einsum
cov_einsum = np.einsum('tm, tmd, tme -> de', W, X, X)

# Calculate cov using for loop
cov_for_loop = np.zeros((D, D))
for t in range(T):
    for m in range(M):
        cov_for_loop += W[t, m] * (X[t, m, :].reshape(-1, 1) @ X[t, m, :].reshape(1, -1))

# Print the results
print("cov (np.einsum):")
print(cov_einsum)
print("cov (for loop):")
print(cov_for_loop)
