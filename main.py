
from loadVars import *
import numpy as np
import subprocess
from sampling import sampling_rlh
from eval import evaluateAero
import pandas as pd
from  makeSurrogate import genNNmodel
from optimization import optimizeGA

lb = np.array([0.0055, 0.3573, 0.0600, -1.0294, 0.3360, -0.071, -0.0486,  -0.02, 0.0, -20.51,   2.5,  0])
ub = np.array([0.0215, 0.6043, 0.1194, -0.3900, 0.5376, -0.057,  0.8204,   0.02, 0.0,      0, 14.74,  4])



# ----- Generating Dataset --------------------------------------------------------------------------- 

# Sampling points generation
n_sampling = 15 * n_var # for LHS, n_sampling = 10 * n_var
X = sampling_rlh(Nsamp=n_sampling, dimen=n_var) # Generate sampling points
X = X*(ub[:] - lb[:]) + lb[:] # Convert the bounds into the real scale


# Evaluating each points``
for i in range (n_sampling):
    evaluateAero(index=i, indi_denorm=X[i,:],fun_name = "transonic_airfoil")

# ----- Iteration ---------------------------------------------------------------------------
for iter in range(1):
    # read dataset --------------------------------------------------------------------------
    df_X = pd.read_csv('.\\Solutions\\dataset.csv', usecols= ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12'])
    X = df_X.to_numpy()
    df_y = pd.read_csv('.\\Solutions\\dataset.csv', usecols= ['cl', 'cd', 'cmy'])
    y = df_y.to_numpy()

    model = genNNmodel(X, y)
    new_dv = optimizeGA(model)
    evaluateAero(new_dv,fun_name = "transonic_airfoil")


        


