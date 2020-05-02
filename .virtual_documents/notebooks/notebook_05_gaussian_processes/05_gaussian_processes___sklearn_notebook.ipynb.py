get_ipython().getoutput("pip install -U scikit-learn")
get_ipython().getoutput("pip install --upgrade pip")


get_ipython().getoutput("pip install --upgrade pip")


import numpy as np
from sklearn import gaussian_process
from matplotlib import pyplot as plt
import matplotlib as mt
mt.rcParams['axes.titlesize'] = 20
mt.rcParams['axes.labelsize'] = 16
mt.rcParams['xtick.labelsize'] = 12
mt.rcParams['ytick.labelsize'] = 12
mt.rcParams['legend.fontsize'] = 14


# The gaussian_process.kernels module has all the builtin
# kernels and also some convinience functions
rbf = gaussian_process.kernels.RBF()
whi = gaussian_process.kernels.WhiteKernel()
exp = gaussian_process.kernels.ExpSineSquared()
dot = gaussian_process.kernels.DotProduct()
rat = gaussian_process.kernels.RationalQuadratic()
mat = gaussian_process.kernels.Matern()
con = gaussian_process.kernels.ConstantKernel()
kernels = [whi, rbf, mat, exp, dot, rat]
kernels


# We can then visualize the kernel's similarity by calling it
x = np.linspace(-1, 1, 100)
x_matrix = x[:, None]  # transforms x in a matrix (len(x) rows and 1 column)
fig, axs = plt.subplots(2, 3, figsize=(12, 9), sharex=True, sharey=True)
for i, kern in enumerate(kernels):
    ax = axs[i // 3, i % 3]
    vals = kern(x_matrix)
    # ax.pcolor(x, x, vals.reshape((len(x), len(x))))
    ax.pcolor(x, x, vals)
    ax.set_title(kern.__class__.__name__)
fig.tight_layout()


# Kernels allow some symbolic math operations like
# summation and exponentiation, allowing us to
# combine basic kernels into new ones
kernel = whi * 0.1 + rbf * 5 + exp * 2 + 0.5 * dot**3
vals = kernel(x_matrix)
plt.figure(figsize=(8, 6))
#plt.pcolor(x, x, vals.reshape((len(x), len(x))))
plt.pcolor(x, x, vals)
plt.title(str(kernel).replace(' +', '\n+'))
plt.colorbar();





# Kernels allow some symbolic math operations like
# summation and exponentiation, allowing us to
# combine basic kernels into new ones
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, ExpSineSquared
kernel = (ConstantKernel(-1) + WhiteKernel() + RBF(length_scale=2.5)) * ExpSineSquared(periodicity=4)
x = np.linspace(-5, 5, 100)
vals = kernel(x[:, None])
plt.figure(figsize=(8, 6))
plt.pcolor(x, x, vals)
plt.title(str(kernel).replace(' +', '\n+'))
plt.colorbar();


get_ipython().getoutput("pip install -U holoviews")


import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern

x = np.linspace(-1, 1, 100)[:, None]

# When run live, this cell's output should match the behavior of the GIF below
similarity1 = lambda *args, **kwargs: similarity(*args, kernel=ExpSineSquared(), **kwargs)
holomap = hv.HoloMap({(length_scale, nu, periodicity):similarity1(length_scale, nu, periodicity) for length_scale in range(0, 5) for nu in range(0, 5) for periodicity in range(0, 5)},  kdims=['length_scale', 'nu', 'periodicity'])
dmap = hv.DynamicMap(similarity1, kdims=['length_scale', 'nu', 'periodicity'])
holomap + dmap


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the monthly dataset
data_monthly = pd.read_csv(
    os.path.join("..", "data", "monthly_in_situ_co2_mlo.csv"), header=56
    )

# - replace -99.99 with NaN
data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)

# fix column names
cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]

data_monthly.columns = cols
cols.remove("--")
cols.remove("--")
data_monthly = data_monthly[cols]

# drop rows with nan
data_monthly.dropna(inplace=True)

# fix time index
data_monthly["day"] = 15
data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
cols.remove("year")
cols.remove("month")
data_monthly = data_monthly[cols]
data_monthly.head()


get_ipython().getoutput("pip install -U pandas")


# function to convert datetimes to numbers that are useful to algorithms
#   this will be useful later when doing prediction

def dates_to_idx(timelist):
    reference_time = pd.to_datetime('1958-03-15')
    t = (timelist - reference_time) / pd.Timedelta(365, "D")
    return np.asarray(t)


t = dates_to_idx(data_monthly.index)

# normalize CO2 levels
y = data_monthly["CO2"].values
first_co2 = y[0]
std_co2 = np.std(y)
y_n = (y - first_co2) / std_co2

data_monthly = data_monthly.assign(t=t)
data_monthly = data_monthly.assign(y_n=y_n)
data_monthly


# split into training and test set
sep_idx = data_monthly.index.searchsorted(pd.to_datetime("2003-12-15"))
data_early = data_monthly.iloc[:sep_idx+1, :]
data_later = data_monthly.iloc[sep_idx:, :]


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.plot(data_monthly.index.values, data_monthly.CO2.values,
        color='b', label='CO2')
ax.plot(data_monthly.index.values, data_monthly.CO2.values,
        color='b', marker='o', linestyle='', alpha=0.2)
ax.axvline(pd.to_datetime("2003-12-15"), linestyle='--', color='k')
ax.legend()
ax.set_title('Raw observed CO2 levels as a function of true time')


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.plot(data_monthly.t.values, data_monthly.y_n.values,
        color='b', label='CO2')
ax.plot(data_monthly.t.values, data_monthly.y_n.values,
        color='b', marker='o', linestyle='', alpha=0.2)
ax.axvline(dates_to_idx(pd.to_datetime("2003-12-15")), linestyle='--', color='k')
ax.legend()
ax.set_title('Rescaled CO2 levels as a function of time index')


from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


X = data_monthly.t.values[:, None]
y = data_monthly.y_n.values
train_X = X[:len(data_early)]
train_y = y[:len(data_early)]
test_X = X[len(data_early):]
test_y = y[len(data_early):]

# To get the raw CO2 levels you should use
# y = data_monthly.CO2.values
# train_y = y[:len(data_early)]
# test_y = y[len(data_early):]


def plot_predictions(gauss_process_regressor,
                     include_observed=True,
                     obs_output=y, obs_input=X):
    pred_y, pred_std = gauss_process_regressor.predict(X, return_std=True)
    plt.figure(figsize=(10, 7))
    x = obs_input[:, 0]  # from 1 column matrix to a simple array
    if include_observed:
        plt.plot(x, obs_output, 'ok', alpha=0.1)
    l, = plt.plot(x, pred_y)
    plt.fill_between(x,
                     pred_y + pred_std,
                     pred_y - pred_std,
                     color=l.get_color(), alpha=0.3)
    if np.allclose(x, data_monthly.t.values):
        plt.axvline(dates_to_idx(pd.to_datetime("2003-12-15")),
                    linestyle='--', color='r')
    else:
        plt.axvline(pd.to_datetime("2003-12-15"), linestyle='--', color='k')


k1 = 50**2 * RBF(length_scale=50.0)  # Long term trend

# Create the regressor
gp0 = GaussianProcessRegressor(kernel=k1, alpha=0.01,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp0


# Train the regressor
gp0.fit(train_X, train_y)


gp0.kernel


# We can see the kernel after training in the kernel_ attribute
gp0.kernel_


# All other parameters of the GP
gp0.get_params()


# The log marginal likelihood of the MAP can be found after training
gp0.log_marginal_likelihood_value_


# The fitted kernel's parameters are packed into an array
gp0.kernel_.theta


# The GP also exposes the log marginal likelihood function callable
k1 = np.linspace(1e-3, 25, 50)
k2 = np.linspace(1e-3, 20, 50)
lml = np.empty((len(k1), len(k2)))
for i, k1_ in enumerate(k1):
    for j, k2_ in enumerate(k2):
        try:
            lml[i, j] = gp0.log_marginal_likelihood([k1_, k2_])
        except ValueError:
            # Numerical instabilities can cause overflows or underflows
            # we replace these errored values with NaN
            lml[i, j] = np.nan


plt.figure(figsize=(9, 7))
plt.pcolor(k2, k1, lml,
           vmin=-100)
plt.colorbar()
plt.plot([gp0.kernel_.theta[1]], [gp0.kernel_.theta[0]],
         'or')
plt.xlabel('k2', fontsize=14)
plt.ylabel('k1', fontsize=14)
plt.title('Log marginal likelihood', fontsize=18)


gp0 = GaussianProcessRegressor(kernel=k1, alpha=0.01,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp0.fit(train_X, train_y)
plot_predictions(gp0)
plt.title(str(gp0.kernel_));


k1 = 50**2 * RBF(length_scale=50.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k4
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.0,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp1.fit(train_X, train_y)
plot_predictions(gp1)
plt.title(str(gp1.kernel_).replace(' +', '\n+'))
gp1.kernel_


k1 = 50**2 * RBF(length_scale=50.0)  # Long term trend
k2 = 20 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=12.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
kernel = k1 + k2
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp2.fit(train_X, train_y)
plot_predictions(gp2)
plt.title(str(gp2.kernel_).replace(' +', '\n+'))
gp2.kernel_


k1 = 50**2 * RBF(length_scale=50.0)
k2 = 20 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k2 + k4
gp3 = GaussianProcessRegressor(kernel=kernel, alpha=0.0,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp3.fit(train_X, train_y)
plot_predictions(gp3)
plt.title(str(gp3.kernel_).replace(' +', '\n+'))
gp3.kernel_


k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k2 + k3 + k4

gp_full = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                   normalize_y=True,
                                   n_restarts_optimizer=3)
gp_full.fit(train_X, train_y)
plot_predictions(gp_full)
plt.title(str(gp_full.kernel_).replace(' +', '\n+'))
gp_full.kernel_


start = pd.to_datetime('2001-01-15')
end = pd.to_datetime('2005-01-15')
sample_range = pd.date_range(start=start,
                             end=end,
                             freq='D')
sample_x = dates_to_idx(sample_range)
data_in_range = data_monthly[start:end]
x_in_range = data_in_range.t.values
y_in_range = data_in_range.y_n.values

gps = [gp1, gp2, gp3, gp_full]
titles = ['gp1', 'gp2', 'gp3', 'gp_full']
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
for i, (gp, title) in enumerate(zip(gps, titles)):
    ax = axs[i // 2, i % 2]
    sample_y = gp.sample_y(sample_x[:, None], n_samples=10)
    ax.plot(sample_range, sample_y, 'b', alpha=0.2)
    ax.plot(data_in_range.index.values, y_in_range, 'ok')
    ax.set_title(title)
    if i // 2 == 1:
        ax.tick_params(axis='x', labelrotation=60)
fig.tight_layout()


# Generate data
train_size = 50
rng = np.random.RandomState(0)
X = rng.uniform(0, 5, 100)[:, None]
y = np.array(X[:, 0] > 2.5, dtype=int)


from matplotlib import pyplot as plt
plt.plot(X, y, 'o')


# Specify Gaussian Processes with fixed and optimized hyperparameters
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = \
    train_test_split(X, y)

gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                   optimizer=None)
gp_fix.fit(X[:train_size], y[:train_size])

gp_opt = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
gp_opt.fit(X[:train_size], y[:train_size])


print("Log Marginal Likelihood (initial): get_ipython().run_line_magic(".3f"", "")
      % gp_fix.log_marginal_likelihood_value_)
print("Log Marginal Likelihood (optimized): get_ipython().run_line_magic(".3f"", "")
      % gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta))

print("Accuracy: get_ipython().run_line_magic(".3f", " (initial) %.3f (optimized)\"")
      % (accuracy_score(y[train_size:], gp_fix.predict(X[train_size:])),
         accuracy_score(y[train_size:], gp_opt.predict(X[train_size:]))))
print("Log-loss: get_ipython().run_line_magic(".3f", " (initial) %.3f (optimized)\"")
      % (log_loss(y[train_size:], gp_fix.predict_proba(X[train_size:])[:, 1]),
         log_loss(y[train_size:], gp_opt.predict_proba(X[train_size:])[:, 1])))


# Plot posteriors
plt.figure(figsize=(9, 7))
plt.scatter(X[:train_size, 0], y[:train_size], c='k', label="Train data",
            edgecolors=(0, 0, 0))
plt.scatter(X[train_size:, 0], y[train_size:], c='g', label="Test data",
            edgecolors=(0, 0, 0))
X_ = np.linspace(0, 5, 100)
plt.plot(X_, gp_fix.predict_proba(X_[:, np.newaxis])[:, 1], 'r',
         label="Initial kernel: get_ipython().run_line_magic("s"", " % gp_fix.kernel_)")
plt.plot(X_, gp_opt.predict_proba(X_[:, np.newaxis])[:, 1], 'b',
         label="Optimized kernel: get_ipython().run_line_magic("s"", " % gp_opt.kernel_)")
plt.xlabel("Feature")
plt.ylabel("Class 1 probability")
plt.xlim(0, 5)
plt.ylim(-0.25, 1.5)
plt.legend(loc="best")


# Plot LML landscape
plt.figure(figsize=(9, 7))
theta0 = np.logspace(0, 8, 30)
theta1 = np.logspace(-1, 1, 29)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
LML = np.array(LML).T
plt.plot(np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1],
         'ko', zorder=10, label='Initial', markersize=12)
plt.plot(np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1],
         'rd', zorder=10, label='Best', markersize=12)
plt.pcolor(Theta0, Theta1, LML)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel("Magnitude")
plt.ylabel("Length-scale")
plt.legend()
plt.title("Log-marginal-likelihood")


# import some data to play with
from sklearn import datasets
from sklearn.gaussian_process.kernels import ConstantKernel
iris = datasets.load_iris()
X = iris.data
y = np.array(iris.target, dtype=int)

isotropic_kernel = 1.0 * RBF(length_scale=1.0)
anisotropic_kernel =  1.0 * RBF(length_scale=np.ones(X.shape[1]))
print('Isotropic kernel = {}. Default parameter array = {}'.
      format(isotropic_kernel, isotropic_kernel.theta))
print('Anisotropic kernel = {}. Default parameter array = {}'.
      format(anisotropic_kernel, anisotropic_kernel.theta))


from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import model_selection
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.array(iris.target, dtype=int)


kernels = [RBF(length_scale=1),
           RBF(length_scale=np.ones(X.shape[1])),
           Matern(length_scale=1),
           Matern(length_scale=np.ones(X.shape[1])),
           DotProduct()]

gps = []
all_scores = []
for kernel in kernels:
    gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5)
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    score = 'precision_macro'
    scores = model_selection.cross_val_score(
        gp,
        X,
        y,
        cv=kf,
        scoring=score,
        n_jobs=-1
    )
    gps.append(gp)
    all_scores.append(scores)
all_scores = np.array(all_scores)
print(all_scores)
best_gp = gps[np.argmax(np.mean(all_scores, axis=1))]
print(best_gp)


print(np.mean(all_scores, axis=1))
print(np.std(all_scores, axis=1))



