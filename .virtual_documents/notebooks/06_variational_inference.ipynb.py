import pyro
import torch
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer import Predictive
import torch.distributions.constraints as constraints
sns.set_style("darkgrid")
figsize=(10,4)
pyro.set_rng_seed(0)


# load data from csv and remove NA values
data = pd.read_csv("data/weatherAUS.csv").dropna()
data.info()


# extract Sydney data 
sydney_data = data.loc[data["Location"]=="Sydney"]

# choose predictors and response for this model
mlr_data = sydney_data[["Rainfall","MinTemp","MaxTemp","Humidity9am",
                        "Humidity3pm","Cloud9am","Cloud3pm"]]

print("n_observations =", len(mlr_data))
mlr_data.head()


# dataset columns to torch tensors
rain = torch.tensor(mlr_data["Rainfall"].values, dtype=torch.float)
predictors = torch.stack([torch.tensor(mlr_data[column].values, dtype=torch.float)
                           for column in ["MinTemp","MaxTemp","Humidity9am",
                                         "Humidity3pm","Cloud9am","Cloud3pm"]], 1)


k = int(0.8 * len(mlr_data))
x_train, y_train = predictors[:k], rain[:k]
x_test, y_test = predictors[k:], rain[k:]

print("x_train.shape =", x_train.shape,"\ny_train.shape =", y_train.shape)
print("\nx_test.shape =", x_test.shape,"\ny_test.shape =", y_test.shape)


# modelling rain in terms of the predictors
def sydney_model(predictors, rain):
    n_observations, n_predictors = predictors.shape

    # sample weights
    w = pyro.sample("w", dist.Normal(torch.zeros(n_predictors), 
                                        torch.ones(n_predictors)))
    b = pyro.sample("b", dist.LogNormal(torch.zeros(1), torch.ones(1)))

    yhat = (w*predictors).sum(dim=1) + b

    # sample observations noise
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.))

    # condition on the observations
    with pyro.plate("rain", len(rain)):
        pyro.sample("obs", dist.Normal(yhat, sigma), obs=rain)


def sydney_guide(predictors, rain=None):
    n_observations, n_predictors = predictors.shape

    w_loc = pyro.param("w_loc", torch.rand(n_predictors), constraint=constraints.positive)
    w_scale = pyro.param("w_scale", torch.rand(n_predictors), 
                         constraint=constraints.positive)

    w = pyro.sample("w", dist.Gamma(w_loc, w_scale))

    b_loc = pyro.param("b_loc", torch.rand(1))
    b_scale = pyro.param("b_scale", torch.rand(1), constraint=constraints.positive)

    b = pyro.sample("b", dist.LogNormal(b_loc, b_scale))


pyro.clear_param_store()

sydney_svi = SVI(model=sydney_model, guide=sydney_guide, 
              optim=optim.ClippedAdam({'lr' : 0.01}), 
              loss=Trace_ELBO()) 

for step in range(2000):
    loss = sydney_svi.step(x_train, y_train)/len(x_train)
    if step % 100 == 0:
        print(f"Step {step} : loss = {loss}")


print("Inferred params:", list(pyro.get_param_store().keys()), end="\n\n")

# w_i and b posterior mean
inferred_w = pyro.get_param_store()["w_loc"]
inferred_b = pyro.get_param_store()["b_loc"]

for i,w in enumerate(inferred_w):
    print(f"w_{i} = {w.item():.8f}")
print(f"b = {inferred_b.item():.8f}")


# print latent params quantile information
def summary(samples):
    stats = {}
    for par_name, values in samples.items():
        marginal = pd.DataFrame(values)
        percentiles=[.05, 0.5, 0.95]
        describe = marginal.describe(percentiles).transpose()
        stats[par_name] = describe[["mean", "std", "5get_ipython().run_line_magic("",", " \"50%\", \"95%\"]]")
    return stats

# define the posterior predictive
predictive = Predictive(model=sydney_model, guide=sydney_guide, num_samples=100,
                        return_sites=("w","b","sigma"))

# get posterior samples on test data
svi_samples = {k: v.detach().numpy() for k, v in predictive(x_test, y_test).items()}

# show summary statistics
for key, value in summary(svi_samples).items():
    print(f"Sampled parameter = {key}\n\n{value}\n")


# compute predictions using the inferred paramters
y_pred = (inferred_w * x_test).sum(1) + inferred_b

print("MAE =", torch.nn.L1Loss()(y_test, y_pred).item())
print("MSE =", torch.nn.MSELoss()(y_test, y_pred).item())


# select Sydney data
sydney_data = data.loc[data["Location"]=="Sydney"]

# replace labels with boolean values
print(sydney_data["RainTomorrow"].unique(), end=" ---> ")
sydney_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)
print(sydney_data["RainTomorrow"].unique())

# classification labels
labels = torch.tensor(sydney_data["RainTomorrow"].values, dtype=torch.double)


# select predictors
df = sydney_data[["Rainfall","Humidity3pm","Cloud3pm","WindSpeed3pm",
                  "Evaporation","Pressure3pm"]]
df.head()


# dataset normalization
df = (df-df.min())/(df.max()-df.min())
df.head()


# torch tensor of features
features = torch.stack([torch.tensor(df[colname].values) for colname in df], dim=1)

# train-test split
k = int(0.8 * len(sydney_data))
x_train, y_train = features[:k], labels[:k]
x_test, y_test = features[k:], labels[k:]

print("x_train.shape =", x_train.shape,"\ny_train.shape =", y_train.shape)
print("\nx_test.shape =", x_test.shape,"\ny_test.shape =", y_test.shape)


# delete previously inferred params from pyro.param_store()
pyro.clear_param_store()

def log_reg_model(x, y):
    n_observations, n_predictors = x.shape
    
    w = pyro.sample("w", dist.Normal(torch.zeros(n_predictors), torch.ones(n_predictors)))
    b = pyro.sample("b", dist.Normal(0.,1.))
    
    # non-linearity
    yhat = torch.sigmoid((w*x).sum(dim=1) + b)
    print(yhat.shape)
    with pyro.plate("data", n_observations):
        # sampling 0-1 labels from Bernoulli distribution
        y = pyro.sample("y", dist.Bernoulli(yhat), obs=y)
    print(y.shape)
        
def log_reg_guide(x, y=None):
    
    n_observations, n_predictors = x.shape
    
    w_loc = pyro.param("w_loc", torch.rand(n_predictors))
    w_scale = pyro.param("w_scale", torch.rand(n_predictors), 
                         constraint=constraints.positive)
    w = pyro.sample("w", dist.Normal(w_loc, w_scale))
    
    b_loc = pyro.param("b_loc", torch.rand(1))
    b_scale = pyro.param("b_scale", torch.rand(1), 
                         constraint=constraints.positive)
    b = pyro.sample("b", dist.Normal(b_loc, b_scale))

    
log_reg_svi = SVI(model=log_reg_model, guide=log_reg_guide, 
              optim=optim.ClippedAdam({'lr' : 0.0002}), 
              loss=Trace_ELBO()) 

losses = []
for step in range(10):
    loss = log_reg_svi.step(x_train, y_train)/len(x_train)
    losses.append(loss)
    if step % 1000 == 0:
        print(f"Step {step} : loss = {loss}")
        
fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("ELBO loss");


w = pyro.get_param_store()["w_loc"]
b = pyro.get_param_store()["b_loc"]

def predict_class(x):
    out = torch.sigmoid((w * x).sum(dim=1) + b)
    return (out>0.5)


correct_predictions = (predict_class(x_test) == y_test).sum().item()

print(f"test accuracy = {correct_predictions/len(x_test)*100:.2f}get_ipython().run_line_magic("")", "")
