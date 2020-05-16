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
figsize = (10, 4)
pyro.set_rng_seed(0)

smoking_data = pd.read_csv("smoking.dat", sep='\s+')
smoking_data.head()


def lists_to_dict(list1, list2=None):
    """Given two lists, zip them and return them as a dict"""
    if list2 is None:
        list2 = [i for i in range(1, len(list1)+1)]
    assert len(list1) == len(list2), "lists are not of the same lengthget_ipython().getoutput(""")
    return dict(zip(list1, list2))


def relabel_values(data_set, columns):
    for column in columns:
        unique_col_values=data_set[column].unique()
        data_set[column].replace(lists_to_dict(unique_col_values), inplace=True)


smoking_raw=copy.deepcopy(smoking_data)
relabel_values(smoking_raw, ["age", "smoke"])
smoking_raw.head()
