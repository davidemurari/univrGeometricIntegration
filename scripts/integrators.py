from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np

def integrate_with_explicit_euler(y0,func,t_eval):
    soln = np.zeros((len(y0),len(t_eval)))
    soln[:,0] = y0
    hs = np.diff(t_eval)
    for i in range(len(t_eval)-1):
        soln[:,i+1] = soln[:,i] + hs[i] * func(t_eval[i],soln[:,i])
    return soln

def implicit_euler_step(t, y, h, func):

    def implicit_eq(y_next):
        # Equation for the implicit method
        return y_next - y - h * func(t + h, y_next)

    # Solve the implicit equation using a root-finding method
    sol = root(implicit_eq, y)
    return sol.x

def integrate_with_implicit_euler(y0, func, t_eval):
    soln = np.zeros((len(y0), len(t_eval)))
    soln[:, 0] = y0
    hs = np.diff(t_eval)
    for i in range(len(t_eval) - 1):
        soln[:, i + 1] = implicit_euler_step(t_eval[i], soln[:, i], hs[i], func)
    return soln

def integrate_with_rk45(y0, func, t_eval, tol=None):
    if not tol==None:
        return solve_ivp(func, t_span=[t_eval[0], t_eval[-1]], t_eval=t_eval, y0=y0, method='RK45', atol=tol).y
    else:
        return solve_ivp(func, t_span=[t_eval[0], t_eval[-1]], t_eval=t_eval, y0=y0, method='RK45').y

def implicit_midpoint_step(t, y, h, func):
    def implicit_eq(y_new):
        # Equation for the implicit midpoint method
        return y_new - y - h * func(t + h/2, (y_new + y)/2)

    # Solve the implicit equation using a root-finding method
    sol = root(implicit_eq, y, tol=1e-15)
    return sol.x

def integrate_with_implicit_midpoint(y0, func, t_eval):
    soln = np.zeros((len(y0), len(t_eval)))
    soln[:, 0] = y0
    hs = np.diff(t_eval)
    for i in range(len(t_eval) - 1):
        soln[:, i + 1] = implicit_midpoint_step(t_eval[i], soln[:, i], hs[i], func)
    return soln
