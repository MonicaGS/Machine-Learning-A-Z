# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:58:08 2018

@author: monica g
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1:].values
