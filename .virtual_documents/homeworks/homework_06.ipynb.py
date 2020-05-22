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
import copy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
figsize = (10, 4)
pyro.set_rng_seed(42)

smoking_data = pd.read_csv("https://data.princeton.edu/wws509/datasets/smoking.dat", sep='\s+')
smoking_data.info()
# convert categorical variables into dummies variables
smoking_data = pd.get_dummies(smoking_data)


# re-scale the populatiopn dependent variable as it is too dominant compared to
# the other ones
smoking_data["pop"] = MinMaxScaler().fit_transform(pd.DataFrame(smoking_data["pop"]))


num_deaths = torch.tensor(smoking_data["dead"].values, dtype=torch.float)
predictors = torch.stack([torch.tensor(smoking_data[column].values, dtype=torch.float)
                         for column in smoking_data.columns if column get_ipython().getoutput("= "dead"], 1)")


X_train, X_test, y_train, y_test = train_test_split(predictors, num_deaths, test_size=0.20, 
                                                    random_state=54, shuffle=True)


def death_model(predictors, deaths):
    n_observations, n_predictors=predictors.shape
    w=pyro.sample("w", dist.Normal(torch.zeros(n_predictors),
                                   torch.ones(n_predictors)))
    b=pyro.sample("b", dist.LogNormal(torch.zeros(1), torch.ones(1)))

    mu_hat=torch.exp((w*predictors).sum(dim=1) + b)

    with pyro.plate("dead", len(deaths)):
        pyro.sample("obs", dist.Poisson(mu_hat), obs = deaths)


def death_guide(predictors, deaths=None):
    n_observations, n_predictors = predictors.shape

    w_loc = pyro.param("w_loc", torch.rand(n_predictors),
                       constraint=constraints.positive)
    w_scale = pyro.param("w_scale", torch.rand(n_predictors), 
                         constraint=constraints.positive)

    w = pyro.sample("w", dist.Gamma(w_loc, w_scale))

    b_loc = pyro.param("b_loc", torch.rand(1))
    b_scale = pyro.param("b_scale", torch.rand(1), constraint=constraints.positive)

    b = pyro.sample("b", dist.LogNormal(b_loc, b_scale))


pyro.clear_param_store()

death_svi = SVI(model=death_model, guide=death_guide, 
                optim=optim.ClippedAdam({'lr' : 0.01}), 
                loss=Trace_ELBO())

for step in range(2000):
    loss = death_svi.step(X_train, y_train)/len(X_train)
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
predictive = Predictive(model=death_model, guide=death_guide, num_samples=100,
                        return_sites=("w","b"))

# get posterior samples on test data
svi_samples = {k: v.detach().numpy() for k, v in predictive(X_test, y_test).items()}

# show summary statistics
for key, value in summary(svi_samples).items():
    print(f"Sampled parameter = {key}\n\n{value}\n")


# compute predictions using the inferred paramters
y_pred = torch.exp((inferred_w * X_test).sum(dim=1) + inferred_b)
print("MAE =", torch.nn.L1Loss()(y_test, y_pred).item())
print("MSE =", torch.nn.MSELoss()(y_test, y_pred).item())


pyro.clear_param_store()
death_guide = pyro.infer.autoguide.AutoMultivariateNormal(death_model)
death_svi = SVI(model=death_model, guide=death_guide, 
                optim=optim.ClippedAdam({'lr' : 0.01}), 
                loss=Trace_ELBO())

for step in range(2000):
    loss = death_svi.step(X_train, y_train)/len(X_train)
    if step % 100 == 0:
        print(f"Step {step} : loss = {loss}")


print("Inferred params:", list(pyro.get_param_store().keys()), end="\n\n")
# w_i and b posterior mean
inferred_w = pyro.get_param_store()["AutoMultivariateNormal.loc"][:-1]
inferred_b = pyro.get_param_store()["AutoMultivariateNormal.loc"][-1]
#inferred_b = pyro.get_param_store()["AutoMultivariateNormal.scale_tril"]
#print(inferred_b.shape)
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
predictive = Predictive(model=death_model, guide=death_guide, num_samples=100,
                        return_sites=("w","b"))

# get posterior samples on test data
svi_samples = {k: v.detach().numpy() for k, v in predictive(X_test, y_test).items()}

# show summary statistics
for key, value in summary(svi_samples).items():
    print(f"Sampled parameter = {key}\n\n{value}\n")


# compute predictions using the inferred paramters
y_pred = torch.exp((inferred_w * X_test).sum(dim=1) + inferred_b)
print("MAE =", torch.nn.L1Loss()(y_test, y_pred).item())
print("MSE =", torch.nn.MSELoss()(y_test, y_pred).item())


get_ipython().run_line_magic("reset", " -f")
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
import copy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
figsize = (10, 4)
pyro.set_rng_seed(42)
from sklearn import datasets


iris = datasets.load_iris()
iris.data = (iris.data - np.min(iris.data))/(np.max(iris.data)-np.min(iris.data))
#iris


iris_predictors = torch.tensor(iris.data, dtype=torch.float)
iris_species = torch.tensor(iris.target, dtype=torch.float)
X_train, X_test, y_train, y_test = train_test_split(iris_predictors, iris_species, test_size=0.20, 
                                                    random_state=42, shuffle=True)



# model and guide functions
def log_reg_model(x, y):
    n_observations, n_predictors = x.shape
    W = [pyro.sample("w_"+str(i), dist.Normal(torch.zeros(n_predictors), torch.ones(n_predictors)))
         for i in range(0, 3)]
    B = [pyro.sample("b_"+str(i), dist.Normal(0., 1.))
         for i in range(0, 3)]
    
    yhat = torch.stack(
        [torch.sigmoid((W[i]*x).sum(dim=1) + B[i])
         for i in range(0, 3)],
        dim=1
    )
    with pyro.plate("data", n_observations):
        # sampling 0-1 labels from Bernoulli distribution
        y = pyro.sample("y", dist.Categorical(yhat), obs=y)
               
def log_reg_guide(x, y=None):
    
    n_observations, n_predictors = x.shape
    
    w_loc_zero = pyro.param("w_loc_zero", torch.rand(n_predictors))
    w_scale_zero = pyro.param("w_scale_zero", torch.rand(n_predictors), constraint=constraints.positive)
    w_zero = pyro.sample("w_0", dist.Laplace(w_loc_zero, w_scale_zero))
    
    b_loc_zero = pyro.param("b_loc_zero", torch.rand(1))
    b_scale_zero = pyro.param("b_scale_zero", torch.rand(1), constraint=constraints.positive)
    b_zero = pyro.sample("b_0", dist.Normal(b_loc_zero, b_scale_zero))
    
    w_loc_one = pyro.param("w_loc_one", torch.rand(n_predictors))
    w_scale_one = pyro.param("w_scale_one", torch.rand(n_predictors), constraint=constraints.positive)
    w_one = pyro.sample("w_1", dist.Laplace(w_loc_one, w_scale_one))
    
    b_loc_one = pyro.param("b_loc_one", torch.rand(1))
    b_scale_one = pyro.param("b_scale_one", torch.rand(1), constraint=constraints.positive)
    b_one = pyro.sample("b_1", dist.Normal(b_loc_one, b_scale_one))
    
    w_loc_two = pyro.param("w_loc_two", torch.rand(n_predictors))
    w_scale_two = pyro.param("w_scale_two", torch.rand(n_predictors), constraint=constraints.positive)
    w_two = pyro.sample("w_2", dist.Laplace(w_loc_two, w_scale_two))
    
    b_loc_two = pyro.param("b_loc_two", torch.rand(1))
    b_scale_two = pyro.param("b_scale_two", torch.rand(1), constraint=constraints.positive)
    b_two = pyro.sample("b_2", dist.Normal(b_loc_two, b_scale_two))



log_reg_svi = SVI(model=log_reg_model, guide=log_reg_guide, 
          optim=optim.ClippedAdam({'lr' : 0.0002}), 
          loss=Trace_ELBO()) 

losses = []
for step in range(10000):
    loss = log_reg_svi.step(X_train, y_train)/len(X_train)
    losses.append(loss)
    if step % 1000 == 0:
        print(f"Step {step} : loss = {loss}")
        
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(losses)
ax.set_title("ELBO loss");


auto_guide = pyro.infer.autoguide.AutoDelta(log_reg_model)
log_reg_svi = SVI(model=log_reg_model, guide=auto_guide, 
          optim=optim.ClippedAdam({'lr' : 0.0002}), 
          loss=Trace_ELBO()) 

losses = []
for step in range(10000):
    loss = log_reg_svi.step(X_train, y_train)/len(X_train)
    losses.append(loss)
    if step % 1000 == 0:
        print(f"Step {step} : loss = {loss}")
        
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(losses)
ax.set_title("ELBO loss");


import re
params_store = pyro.get_param_store()
pattern_w = re.compile('w_loc_*')
pattern_b = re.compile('b_loc_*')
#get the loc values according to the string patterns defined above
W = [params_store[key] for key in params_store.keys() if pattern_w.match(key)]
B = [params_store[key] for key in params_store.keys() if pattern_b.match(key)]

# this function returns the class that achievied the maximum probability 
# over the others
def predict_class(predictors, param_W, param_B):
    result = torch.stack(
        [torch.sigmoid((W[i] * predictors).sum(dim=1) + B[i])
         for i in range(0, len(W))],
        dim=1
    )
    return torch.argmax(result, dim=1)

predictions = predict_class(X_test, W, B)
predictions_vs_real = list(zip(predictions , y_test))

def class_checker(class_label):
    def func(predicted_vs_real):
        return predicted_vs_real[0] == predicted_vs_real[1] == class_label
    return func


count_zero = len(list(filter(class_checker(0), predictions_vs_real)))
count_one = len(list(filter(class_checker(1), predictions_vs_real)))
count_two = len(list(filter(class_checker(2), predictions_vs_real)))

correct_pred_zero = (count_zero / len(y_test[y_test == 0])) * 100
correct_pred_one = (count_one / len(y_test[y_test == 1])) * 100
correct_pred_two = (count_two / len(y_test[y_test == 2])) * 100

correct_pred_final = (predictions == y_test).sum().item()

print("Accuracy for class 0 {:.2f}".format(correct_pred_zero),"get_ipython().run_line_magic("")", "")
print("Accuracy for class 1 {:.2f}".format(correct_pred_one),"get_ipython().run_line_magic("")", "")
print("Accuracy for class 2 {:.2f}".format(correct_pred_two),"get_ipython().run_line_magic("")", "")

print(f"Overal test accuracy = {correct_pred_final/len(X_test)*100:.2f}get_ipython().run_line_magic("")", "")
