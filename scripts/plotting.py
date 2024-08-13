import matplotlib.pyplot as plt
import numpy as np

def get_plots(soln,t_eval,method):
    plt.rcParams["figure.figsize"] = (18, 4)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 15

    plt.subplot(1, 3, 1)
    plt.plot(t_eval,soln[0],'r-')
    plt.plot(t_eval[0],soln[0,0],'r-',label=r"$x$")
    plt.plot(t_eval,soln[1],'k-')
    plt.plot(t_eval[0],soln[0,1],'k-',label=r"$v$")
    plt.xlabel(r"$t$")
    plt.legend()


    plt.subplot(1, 3, 2, aspect="equal")
    plt.plot(soln[0],soln[1],'k-',linewidth=3)
    
    x_circle = np.cos(t_eval)
    y_circle = np.sin(t_eval)
    plt.plot(x_circle,y_circle,'c-',linewidth=1)
    
    plt.plot(x_circle[0],y_circle[0],'c-',linewidth=1,label="Correct energy level")
    plt.plot(soln[0,0],soln[1,0],'ro',label="Initial condition")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$")
    plt.legend()

    plt.subplot(1, 3, 3)
    E = np.linalg.norm(soln,axis=0,ord=2)**2 # Norm along the solution
    E0 = np.linalg.norm(soln[:,0],ord=2)**2 # Initial norm squared
    plt.semilogy(t_eval,np.abs(E0 - E),'k-',linewidth=0.5)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\left|E(x(t),v(t))-E(x_0,v_0)\right|$")

    plt.suptitle(f"Results with {method}")

    plt.show();