"""
    The kLGrowth package allows to reproduce three-dimensional
    diagrams of the growth model with a k-logistic production
    function.
    These diagrams can be reproduced both in matplotlib and in
    plotly.
"""
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Parameters:
    """
    An object of the class Parameters stores the parameter
    values of the model.
    The default values are those of scenario 10 (cfr. the
    article related to this package).
    
    :ivar M: the upper bound of the production function
    :vartype M: (non-negative) float
    :ivar kc: the inflection point of the production function
    :vartype kc: float
    :ivar k: the coeffient of the expk function
    :vartype k: float
    :ivar sr: the saving rate for shareholders
    :vartype sr: float (between 0.0 and 1.0)
    :ivar sw: the saving rate for workers
    :vartype sw: float (between 0.0 and 1.0)
    :ivar d: the capital depreciation rate delta
    :vartype delta: float (between 0.0 and 1.0)
    :ivar n: the labor force growth rate
    :vartype n: (non-negative) float
    :ivar TRANSIENT: number of iterations considered before the
        steady state
    :vartype TRANSIENT: integer
    :ivar REGIME: number of iterations considered in the steady
        state
    :vartype REGIME: integer
    :ivar tol: tolerance when evaluating the orbits
    :vartype tol: float
    """
    M = 6
    kc = 6
    k = 0.8
    sr = 1
    sw = .2
    d = .2
    n = .5
    TRANSIENT = 2000
    REGIME = 500
    tol = 1e-5


def expk(x, k):
    """
    expk: the k-exponential function

    :param x: the argument of the k-exponential function
    :type x: float (or array of floats)
    :param k: the k coeffient of the k-exponential function
    :type k: float
    :returns: expk(x)
    :rtype: float (or array of floats)
    """
    if k == 0.0:
        y = np.exp(x)
    else:
        y = (np.sqrt(1 + k**2*x**2) + k*x)**(1/k)
    return y


def sigmak(x, k):
    """
    sigmak: the sigmak function

    :param x: the argument of the sigmak function
    :type x: float (or array of floats)
    :param k: the k coeffient of the sigmak function
    :type k: float
    :returns: sigmak(x)
    :rtype: float (or array of floats)
    """
    y = 1/(1 + expk(-x, k))
    return y


def sigmak1(x, k):
    """
    sigmak1: this function computes the first derivative of the
    sigmak function with respect to its argument x

    :param x: the argument of the sigmak1 function
    :type x: float (or array of floats)
    :param k: the k coeffient of the sigmak function
    :type k: float
    :returns: the first derivative of the
    sigmak function with respect to its argument x
    :rtype: float (or array of floats)
    """
    y = sigmak(x, k)*(1 - sigmak(x, k))/\
        np.sqrt(1 + k**2*x**2)
    return y


def sigmak2(x, k):
    """
    sigmak2: this function computes the second derivative of the
    sigmak function with respect to its argument x

    :param x: the argument of the sigmak1 function
    :type x: float (or array of floats)
    :param k: the k coeffient of the sigmak function
    :type k: float
    :returns: the second derivative of the
    sigmak function with respect to its argument x
    :rtype: float (or array of floats)
    """
    y = ((1 - 2*sigmak(x, k))/np.sqrt(1 + k**2*x**2) - \
         (k**2*x)/(1 + k**2*x**2))*sigmak1(x, k)
    return y


def f(par, kt):
    """
    f: this function computes the production function evaluated
    at kt

    :param par: an object of the Parameter class
    :type par: Parameter
    :param kt: the capital per worker where the production
    function is evaluated
    :type kt: float
    :returns: the production function evaluated at kt and with
    the parameter values stored in par
    :rtype: float
    """
    M = par.M
    kc = par.kc
    k = par.k
    y = M*(sigmak(kt - kc, k) - sigmak(-kc, k))/\
        (1 - sigmak(-kc, k))
    return y


def f1(par, kt):
    """
    f1: this function computes the first derivative of the
    production function evaluated at kt

    :param par: an object of the Parameter class
    :type par: Parameter
    :param kt: the capital per worker where the production
    function is evaluated
    :type kt: float
    :returns: the first derivative of the production function
    evaluated at kt and with the parameter values stored in par
    :rtype: float
    """
    M = par.M
    kc = par.kc
    k = par.k
    y = M*sigmak1(kt - kc, k)/(1 - sigmak(-kc, k))
    return y


def f2(par, kt):
    """
    f2: this function computes the first derivative of the
    production function evaluated at kt

    :param par: an object of the Parameter class
    :type par: Parameter
    :param kt: the capital per worker where the production
    function is evaluated
    :type kt: float
    :returns: the first derivative of the production function
    evaluated at kt and with the parameter values stored in par
    :rtype: float
    """
    M = par.M
    kc = par.kc
    k = par.k
    y = M*sigmak2(kt - kc, k)/(1 - sigmak(-kc, k))
    return y


def F(par, kt):
    """
    F: the map of the growth model

    :param par: an object of the Parameter class
    :type par: Parameter
    :param kt: the capital per worker where the map is evaluated
    :type kt: float
    :returns: the map evaluated at kt and with the parameter
    values stored in par
    :rtype: float
    """
    M = par.M
    kc = par.kc
    k = par.k
    n = par.n
    d = par.d
    sr = par.sr
    sw = par.sw
    y = 1/(1 + n)*((1 - d)*kt + sw*f(par, kt) + \
                   (sr - sw)*kt*f1(par, kt))
    return y


def F1(par, kt):
    """
    F1: the first derivative of the map of the growth model

    :param par: an object of the Parameter class
    :type par: Parameter
    :param kt: the capital per worker where the map is evaluated
    :type kt: float
    :returns: the first derivative of the map evaluated at kt
    and with the parameter values stored in par
    :rtype: float
    """
    M = par.M
    kc = par.kc
    k = par.k
    n = par.n
    d = par.d
    sr = par.sr
    sw = par.sw
    y = 1/(1 + n)*(1 - d + sr*f1(par, kt) + \
        (sr - sw)*kt*f2(par, kt))
    return y


def generate_df_3D_M_kc_k(par, M_min, M_max, kc_min, kc_max,
                          k_min, k_max, step_M, step_kc, step_k,
                          k0, fOut = None):
    """
    generate_df_3D_M_kc_k: this function creates a dataframe
    where each row reports a combination of the three
    parameters M, kc, and k, together with the period of the
    cycles at the steady state. All the remaining parameters
    are fixed to the values stored in the object par passed
    as a parameter.
    The period value is set to:
        - 1: for cycles of period 1
        - 2: for cycles of period 2
        - 3: for cycles of period 3
        - 4: for cycles of period 4
        - 5: for cycles of period 5
        - 6: for cycles of period 6
        - 7: for cycles of period 8
        - 8: for cycles of period 16
        - 9: for cycles of period 32
        - 10: for all the remaining cases.
        

    :param par: an object of the Parameter class
    :type par: Parameter
    :param M_min: the minimum value of parameter M considered
    :type M_min: float
    :param M_max: the maximum value of parameter M considered
    :type M_max: float
    :param kc_min: the minimum value of parameter kc considered
    :type kc_min: float
    :param kc_max: the maximum value of parameter kc considered
    :type kc_max: float
    :param k_min: the minimum value of parameter k considered
    :type k_min: float
    :param k_max: the maximum value of parameter k considered
    :type k_max: float
    :param step_M: the step of parameter M ranging from M_min
    to M_max
    :type step_M: float
    :param step_kc: the step of parameter kc ranging from kc_min
    to kc_max
    :type step_kc: float
    :param step_k: the step of parameter k ranging from k_min
    to k_max
    :type step_k: float
    :param k0: the initial seed of the dynamics considered
    varying the three parameters M, kc, and k
    :type k0: float
    :param fOut: optional parameter containing the path where
    to save the produced dataframe
    :type fOut: string
    :returns: a dataframe where each row reports a combination
    of the three parameters M, kc, and k, together with the
    period of the cycles at the steady state.
    :rtype: (Pandas) Dataframe
    """
    M_orig = par.M
    kc_orig = par.kc
    k_orig = par.k
    M_values = np.arange(M_min, M_max + step_M, step_M)
    kc_values = np.arange(kc_min, kc_max + step_kc, step_kc)
    k_values = np.arange(k_min, k_max + step_k, step_k)   
    df = pd.DataFrame(columns = ["M", "k_c", "k", "y"])
    for M in M_values:
        par.M = M
        for kc in kc_values:
            par.kc = kc
            print(k0, M, kc)
            for k in k_values:
                par.k = k
                kt = k0
                y = []
                for _ in range(par.TRANSIENT):
                    kt = F(par, kt)
                for _ in range(par.REGIME):
                    kt = F(par, kt)
                    if kt < par.tol:
                        y.append(0.0)
                    else:
                        y.append(kt)
                if abs(y[-1] - y[-2]) < par.tol:
                    val = 1
                elif abs(y[-1] - y[-3]) < par.tol:
                    val = 2
                elif abs(y[-1] - y[-4]) < par.tol:
                    val = 3
                elif abs(y[-1] - y[-5]) < par.tol:
                    val = 4
                elif abs(y[-1] - y[-6]) < par.tol:
                    val = 5
                elif abs(y[-1] - y[-7]) < par.tol:
                    val = 6
                elif abs(y[-1] - y[-9]) < par.tol:
                    val = 7
                elif abs(y[-1] - y[-17]) < par.tol:
                    val = 8
                elif abs(y[-1] - y[-33]) < par.tol:
                    val = 9
                else:
                    val = 10
                newRow = pd.DataFrame([{"M": M, "k_c": kc,
                                        "k": k,
                                        "y": val}])
                df = pd.concat([df, newRow], ignore_index = True)
    if fOut is not None:
        df.to_excel(fOut, index = False)
    par.k = k_orig
    par.kc = kc_orig
    par.M = M_orig
    return df


def format_string(s):
    """
    format_string: this function transform the string s passed
    as a parameter to a formatted string in latex style to
    appear as an axis label in a matplotlib figure.

    :param s: string equal to "M", or "k_c", or "k".
    If s is not equal to any of these string, the output is
    the null string ""
    :type s: string
    :returns: a formatted string in latex style to appear as an
    axis label in a matplotlib figure
    :rtype: string
    """
    if s == 'M':
        auxString = r"$M$"
    elif s == "k_c":
        auxString = r"$k_c$"
    elif s == 'k':
        auxString = r"$\kappa$"
    else:
        auxString = ""
    return auxString


def plot_diagram(df, mat, limits = None):
    """
    plot_diagram: this function calls appropriate subfunctions
    in order to make a three-dimensional diagram.
    It is possible to limit the range of the parameters by
    passing the optional parameter limits.
    The parameter mat specifies whether the diagram has to be
    plotted in matplotlib or plotly

    :param df: dataframe where each row reports a combination
    of the three parameters M, kc, and k, together with the
    period of the cycles at the steady state.
    :type df: (Pandas) Dataframe
    :param mat: boolean parameter. If true, it calls the
    subfunction that plots the three-dimensional diagram in
    matplotlib. Otherwise, it calls the subfunction that plots
    the three-dimensional diagram in plotly
    :type mat: boolean
    :param limits: optional parameter consisting of a dictionary
    with the limits of the parameters. As an example, if the
    limits are 20.0 <= M <= 60.0 and $10.0 <= kc <= 30.0,
    the dictionary will be:
        {"M": [20, 60], "k_c": [10.0, 30.0]}
    :type lim: dict
    """
    if limits is not None:
        for param in limits.keys():
            lb = limits[param][0]
            df = df.loc[df[param] >= lb]
            ub = limits[param][1]
            df = df.loc[df[param] <= ub]
    if mat:
        print("plot_diagram_3D_mat")
        plot_diagram_3D_mat(df)
    else:
        print("plot_diagram_3D")
        plot_diagram_3D(df)


def plot_diagram_3D(df):
    """
    plot_diagram_3D: this function plots a the three-dimensional
    diagram in plotly according to the dataframe df.
    
    :param df: dataframe where each row reports a combination
    of the three parameters M, kc, and k, together with the
    period of the cycles at the steady state.
    :type df: (Pandas) Dataframe
    """
    x_header = df.columns[0]
    y_header = df.columns[1]
    z_header = df.columns[2]
    val_header = df.columns[-1]
    lab = {1: "1", 2: "2", 3: "3", 4: "4",
           5: "5", 6: "6", 7: "8", 8: "16",
           9: "32", 10: "Other"}
    df['Period'] = df['y'].map(lab)
    df = df.sort_values(by = ['y'], ascending = True)
    fig = px.scatter_3d(df, x = x_header, y = y_header,
                        z = z_header, color = "Period")
    fig.show()


def plot_diagram_3D_mat(df):
    """
    plot_diagram_3D: this function plots a the three-dimensional
    diagram in matplotlib according to the dataframe df.
    
    :param df: dataframe where each row reports a combination
    of the three parameters M, kc, and k, together with the
    period of the cycles at the steady state.
    :type df: (Pandas) Dataframe
    """
    x_header = df.columns[0]
    y_header = df.columns[1]
    z_header = df.columns[2]
    val_header = df.columns[-1]
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    scatter = ax.scatter(df[x_header], df[y_header],
                         df[z_header],
                         c = df[val_header],
                         cmap="viridis")
    handles, labels = scatter.legend_elements()
    labels2 = ['$\\mathdefault{1}$',
               '$\\mathdefault{2}$',
               '$\\mathdefault{3}$',
               '$\\mathdefault{4}$',
               '$\\mathdefault{5}$',
               '$\\mathdefault{6}$',
               '$\\mathdefault{8}$',
               '$\\mathdefault{16}$',
               '$\\mathdefault{32}$',
               'Other']
    ax.legend(handles, labels2, title="Period",
              loc='center left', bbox_to_anchor=(1, 0.5))
    x_header = format_string(x_header)
    y_header = format_string(y_header)
    z_header = format_string(z_header)
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    ax.set_zlabel(z_header)
    fig.tight_layout()
    fig.show()
