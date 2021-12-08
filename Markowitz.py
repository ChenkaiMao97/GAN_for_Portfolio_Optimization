import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
# # n_assets = 4
# k = 15
# n_obs = 40
# returns = np.random.randn(k, n_obs)

N = 0
num_stocks = 22
b = 40
f = 20

fake_batches = np.load("portfolio_data/fake_batches.npy")[:,0]
real_batch = np.load("portfolio_data/real_batches.npy")[0]
scales = np.load("portfolio_data/scales.npy")[0]

returns = real_batch[:,1:b+1]/real_batch[:,:b] -1
returns = returns.astype(np.double)

def get_mu_sigma(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    #desired returns
    mus = [10**(3 * t/N - 1.0) for t in range(N)]

    # mus = np.array([1] + [1 + (t - 1)*(rmax - 1)/(Z-1) for t in range(2, 25)]) -1
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns)*40)
    pbar = opt.matrix(np.mean(returns, axis=1))

    print("S:", S)
    print("pbar:", pbar)

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    np.save("Mkreturns.npy", returns)
    np.save("Mkrisks.npy", risks)
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    # return np.asarray(wt), returns, risks
    return portfolios, returns, risks


weights, returns, risks = get_mu_sigma(returns)
print(len(weights))
np.save("M_portfolio.npy", np.stack(weights))
np.save("risks.npy", np.stack(risks))

fig = plt.figure()
plt.plot(risks, returns)
plt.show()
print(weights)