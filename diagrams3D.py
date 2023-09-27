"""
    diagrams3D: demo Python file for generating
    three-dimensional diagrams of the growth model in
    matplotlib and plotly.
"""
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import kLGrowth as kl


# The parameters are set to the values of scenario 10
par = kl.Parameters()
par.M = 10
par.kc = 10
par.k = 0.2
par.sr = .7
par.sw = .1
par.d = .2
par.n = .5

# Maximum and minimum points in scenario 10
k_M = 10.3109130859375
k_m = 14.510421752929688

# Generate a dataframe with the dynamics starting from k_m
df = kl.generate_df_3D_M_kc_k(par, 10, 15, 0, 15, 0, 1, 1, 1,
                              .1, k_m, "demo.xlsx")

# 3D diagram in matplotlib with some limits on the parameters
lim = {"k_c": [6, 12], "k": [0, 0.5]}
kl.plot_diagram(df, True, lim)

# 3D diagram in plotly without limits on the parameters
kl.plot_diagram(df, False)
