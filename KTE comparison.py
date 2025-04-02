import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from doubleml import DoubleMLData, DoubleMLPLR
import math
from scipy.stats import norm

np.random.seed(123)

# ==== Core Functions ====
def delta_0(x, theta):
    return abs(theta + 1.0) * (np.sin(theta * x) + np.cos(theta * x))

def f_perfect(x, theta, N):
    return x * theta + (1.0 / math.sqrt(N)) * delta_0(x, theta)

def g_function(z):
    return np.exp(z / 10.0) * np.sin(z)

def generate_data(n=500, theta_true=1.0, sigma=0.2, rho=1):
    E1 = np.random.rand(n)
    E2 = np.random.rand(n)
    E3 = np.random.rand(n)
    X = (E1 + rho * E2) / (1 + rho)
    Z = (E3 + rho * E2) / (1 + rho)
    fx = f_perfect(X, theta_true, n)
    gz = g_function(Z)
    noise = np.random.normal(0, sigma, size=n)
    Y = fx + gz + noise
    return X, Z, Y

# ==== KTE ====
def kte_cross_fitting(X, Z, Y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    theta_values = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Z_train, Z_test = Z[train_idx], Z[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        g_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)
        g_model.fit(Z_train.reshape(-1, 1), Y_train)
        Y_resid_train = Y_train - g_model.predict(Z_train.reshape(-1, 1))

        r_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)
        r_model.fit(Z_train.reshape(-1, 1), X_train)
        X_resid_train = X_train - r_model.predict(Z_train.reshape(-1, 1))

        W_train = X_resid_train
        W_test = X_test - r_model.predict(Z_test.reshape(-1, 1))

        f_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)
        f_model.fit(W_train.reshape(-1, 1), Y_resid_train)

        h_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)
        h_model.fit(W_train.reshape(-1, 1), W_train)

        f_pred = f_model.predict(W_test.reshape(-1, 1))
        h_pred = h_model.predict(W_test.reshape(-1, 1))

        numerator = np.sum(f_pred * h_pred)
        denominator = np.sum(X_test * h_pred)
        theta_fold = numerator / denominator if abs(denominator) > 1e-12 else 0.0
        theta_values.append(theta_fold)
    return np.mean(theta_values)

# ==== DML ====
def dml_cross_fitting(X, Z, Y, Q=5):
    kf = KFold(n_splits=Q, shuffle=True, random_state=42)
    theta_estimates = []
    for train_idx, test_idx in kf.split(X):
        X_train, Z_train, Y_train = X[train_idx], Z[train_idx], Y[train_idx]
        X_train_2d = X_train.reshape(-1, 1)
        Z_train_2d = Z_train.reshape(-1, 1)
        dml_data = DoubleMLData.from_arrays(Z_train_2d, Y_train, X_train_2d)

        ml_l = LinearRegression()  # intentionally underpowered
        ml_m = LinearRegression()
        dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=2)
        dml_plr.fit()
        theta_estimates.append(dml_plr.coef[0])
    return np.mean(theta_estimates)

# ==== Simulation ====
n_sims = 20
theta_true = 1.0
theta_kte_all = []
theta_dml_all = []

for sim in range(n_sims):
    X, Z, Y = generate_data(n=500, theta_true=theta_true, sigma=1.0, rho=1)
    theta_kte_all.append(kte_cross_fitting(X, Z, Y))
    theta_dml_all.append(dml_cross_fitting(X, Z, Y))

# ==== Plotting ====
for sim in range(n_sims):
    X, Z, Y = generate_data(n=500, theta_true=theta_true, sigma=1.0, rho=1)

    theta_kte = kte_cross_fitting(X, Z, Y)
    theta_dml, _ = dml_cross_fitting(X, Z, Y)

    theta_kte_all.append(theta_kte)
    theta_dml_all.append(theta_dml)

# ==== Plotting ====
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, data, label in zip(
    axs,
    [theta_kte_all, theta_dml_all],
    ["KTE (including $\\delta_0$)", "DML (ignoring $\\delta_0$)"]
):
    errors = np.array(data) - theta_true

    # Histogram
    ax.hist(errors, bins=30, density=True, color='sandybrown', label='Simulation')

    # Normal overlay
    x_vals = np.linspace(errors.min()-0.5, errors.max()+0.5, 300)
    std_hat = np.std(errors, ddof=1)
    ax.plot(x_vals, norm.pdf(x_vals, 0, std_hat), label='N(0, $\\hat{\\Sigma}$)', lw=2)

    ax.set_title(label)
    ax.set_xlabel(r"$\hat{\theta} - \theta_0$")
    ax.set_ylabel("Probability density function")
    ax.legend()

plt.suptitle("Empirical Distribution of Estimator Error (KTE vs DML)", fontsize=14)
plt.tight_layout()
plt.show()






