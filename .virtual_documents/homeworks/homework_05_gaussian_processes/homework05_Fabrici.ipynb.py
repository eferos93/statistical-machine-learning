from sklearn import preprocessing
import os
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


covid19 = pd.read_csv("COVID_national_20200421.csv")
covid19_cleaned = covid19.drop(
    ["ICU", "Unnamed: 0", "new_infections", "hospitalized", "cumulative_infections", "recovered", "quarantined"],
    axis='columns')
covid19_cleaned['date'] = pd.to_datetime(covid19_cleaned['date'])
covid19_cleaned.info()
covid19_cleaned.set_index('date', inplace=True)
covid19_cleaned.head()


def normalise_column(col):
    """
    Normalisation of the given vector
    @param col Columns to be normalised
    @return Normalised column
    """
    return (col - col[0]) / np.std(col)


covid19_normalised = covid19_cleaned

covid19_normalised = covid19_normalised.apply(normalise_column)
covid19_normalised.tail()


def plot_data(x_data, y_data, label_function: str, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    ax.plot(x_data, y_data,
            color='b', label=label_function)
    ax.plot(x_data, y_data,
            color='b', marker='o', linestyle='', alpha=0.2)
    ax.axvline(pd.to_datetime("2020-04-14"), linestyle='--', color='k')
    ax.legend()
    ax.set_title(title)


def plot_predictions(gauss_process_regressor, label: str,
                     obs_output, obs_input, x_label=covid19_normalised.index.values,
                     include_observed=True):
    pred_y, pred_std = gauss_process_regressor.predict(obs_input, return_std=True)
    plt.figure(figsize=(15, 9))
    if include_observed:
        plt.plot(x_label, obs_output, 'ok', alpha=0.1, label=label+" observed", color='r')
    l, = plt.plot(x_label, pred_y, label=label+" prediction")
    plt.fill_between(x_label,
                     pred_y + pred_std,
                     pred_y - pred_std,
                     color=l.get_color(), alpha=0.3)
    plt.axvline(pd.to_datetime("2020-04-14"), linestyle='--', color='k')
    plt.legend()


plot_data(covid19_cleaned.index.values, covid19_cleaned.swabs.values,
          'Cumulative swabs',
          'Cumulative number of swabs in Italy')


def split_data_set(data_set, output_label: str, index=pd.to_datetime("2020-04-14")):
    sep_idx = data_set.index.searchsorted(index)
    data_early = data_set.iloc[:sep_idx+1, :]
    # data_later = data_set.iloc[sep_idx:, :]
    X = data_set.drop(output_label, axis=1).to_numpy()
    y = data_set[output_label].values
    X_train = X[:len(data_early), :]
    y_train = y[:len(data_early)]
    X_test = X[len(data_early):, :]
    y_test = y[len(data_early):]
    # return {'train_set': (X_train, y_train), 'test_set': (X_test, y_test), 'full_set': (X, y)}
    return X_train, y_train, X_test, y_test, X, y


X_train, y_train, X_test, y_test, X, y = split_data_set(covid19_normalised, 'swabs')

k1 = 50**2*RBF(length_scale=50.0)  # Long term trend

# Create the regressor
gp0 = GaussianProcessRegressor(kernel=k1, alpha=0.01,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp0.fit(X_train, y_train)

plot_predictions(gp0, "Cumulative swabs", y, X,)


k1 = 50**2 * RBF(length_scale=50.0)
k4 = 1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.5**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k4
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                               normalize_y=False,
                               n_restarts_optimizer=3)
gp1.fit(X_train, y_train)
plot_predictions(gp1, 'Cumulative swabs', y, X)


k1 = 50**2 * RBF(length_scale=50.0)
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k3 + k4
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                               normalize_y=False,
                               n_restarts_optimizer=3)
gp1.fit(X_train, y_train)
plot_predictions(gp1, 'Cumulative swabs', y, X)


k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
# medium term irregularities
k3 = 50**2 * RationalQuadratic(length_scale=10.0, alpha=10.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k3 + k4

gp_full = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                                   normalize_y=False,
                                   n_restarts_optimizer=3)

gp_full.fit(X_train, y_train)
plot_predictions(gp_full, 'Cumulative swabs', y, X)


covid19_daily_swabs = covid19_cleaned
# del covid19_daily_swabs
# covid19_daily_swabs = covid19_cleaned
# first element after diff() will be NaN, 
# thus I substitute it with the actual value
covid19_daily_swabs['swabs'] = covid19_daily_swabs['swabs'].diff().fillna(4324)


_ = covid19_daily_swabs.plot()


covid19_daily_swabs = covid19_daily_swabs.apply(normalise_column)
X_train, y_train, X_test, y_test, X, y = split_data_set(covid19_daily_swabs, 'swabs')


k1 = 50**2*RBF(length_scale=50.0)  # Long term trend

# Create the regressor
gp0 = GaussianProcessRegressor(kernel=k1, alpha=0.01,
                               normalize_y=True,
                               n_restarts_optimizer=3)
gp0.fit(X_train, y_train)

plot_predictions(gp0, "Daily swabs", y, X)


k1 = 50**2 * RBF(length_scale=50.0)
k4 = 1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.5**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k4
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                               normalize_y=False,
                               n_restarts_optimizer=3)
gp1.fit(X_train, y_train)
plot_predictions(gp1, 'Daily swabs', y, X)
plt.savefig('daily_swabs.png')


k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
# medium term irregularities
k3 = 50**2 * RationalQuadratic(length_scale=10.0, alpha=10.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, 1e9))  # noise terms
kernel = k1 + k3 + k4

gp_full = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                                   normalize_y=False,
                                   n_restarts_optimizer=3)

gp_full.fit(X_train, y_train)
plot_predictions(gp_full, 'Daily swabs', y, X)


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

gp_full = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                                   normalize_y=True,
                                   n_restarts_optimizer=3)

gp_full.fit(X_train, y_train)
plot_predictions(gp_full, 'Daily swabs', y, X)
plt.savefig('daily_swabs.png')


from sklearn.gaussian_process.kernels import DotProduct
import sklearn.gaussian_process.kernels as kernels
k_dot = DotProduct()

gp = GaussianProcessRegressor(kernel=k_dot, alpha=0.01,
                              normalize_y=True,
                              n_restarts_optimizer=3)

gp.fit(X_train, y_train)
plot_predictions(gp, 'Daily swabs', y, X)
plt.savefig('plt_daily swabs.png')


k_dot_v2 = kernels.Exponentiation(k_dot, 1.5)

gp = GaussianProcessRegressor(kernel=k_dot_v2, alpha=0.01,
                              normalize_y=True,
                              n_restarts_optimizer=3)

gp.fit(X_train, y_train)
plot_predictions(gp, 'Daily swabs', y, X)


k1 = 1000**2 * RBF(length_scale=1.0)
k2 = 1**(1/2)*kernels.ExpSineSquared(periodicity=1.5, length_scale=10)
k_dot = DotProduct(sigma_0=1)
kernel = k1 + 50**10*k_dot + k2

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                              normalize_y=True,
                              n_restarts_optimizer=3)
gp.fit(X_train, y_train)
plot_predictions(gp, 'Daily swabs', y, X)
plt.savefig('plt_daily_swabs.png')


k3 = 50**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k1 = 1000**2 * RBF(length_scale=1.0)
k2 = 1**(1/2)*kernels.ExpSineSquared(periodicity=1.5, length_scale=10)
k_dot = 50**10*DotProduct(sigma_0=1)
kernel = k1 + k_dot + k2

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                              normalize_y=True,
                              n_restarts_optimizer=3)
gp.fit(X_train, y_train)
plot_predictions(gp, 'Daily swabs', y, X)
plt.savefig('plt_daily_swabs_v2.png')
