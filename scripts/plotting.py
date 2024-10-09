import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
from scripts.utilsLotkaVolterra import H as FI_LotkaVolterray
from scripts.utilsLotkaVolterra import alpha,beta,gamma,delta
plt.rcParams["figure.figsize"] = (18, 4)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15

def get_plots(soln,t_eval,method):

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

def plot_energy_behaviour_frb(timespan,sol_EE,sol_IMP,inertia_matrix):
    #This method aims to plot the energy behaviour for the free rigid body

    y0 = sol_EE[:,0] #Initial condition
    h = timespan[1] - timespan[0] #time step
    N = len(timespan)-1

    I1 = inertia_matrix[0,0]
    I2 = inertia_matrix[1,1]
    I3 = inertia_matrix[2,2]

    E1 = lambda y : y[0]**2+y[1]**2+y[2]**2
    E2 = lambda y : y[0]**2/I1+y[1]**2/I2+y[2]**2/I3

    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1],hspace=0.5)

    ax1 = fig.add_subplot(gs[:, 0], projection='3d')  # 1 row, 2 cols, 1st subplot

    ax1.plot(sol_IMP[0], sol_IMP[1], sol_IMP[2], 'b-', label='Implicit Mid-Point')
    ax1.plot(sol_EE[0], sol_EE[1], sol_EE[2], 'k--', label='Explicit Euler')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x, y, z, color='r', alpha=0.1)
    ax1.set_xlabel(r'$x_1$',fontsize=20)
    ax1.set_ylabel(r'$x_2$',fontsize=20)
    ax1.set_zlabel(r'$x_3$',fontsize=20)

    plt.legend()

    ax1.set_title("Comparison of the solutions",fontsize=20)

    elev = 30  # Example elevation angle
    azim = 130  # Example azimuth angle

    ax1.view_init(elev=elev, azim=azim)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='z', labelsize=15)

    ax1.axis('equal')

    ax2_0 = fig.add_subplot(gs[0,1])  # 1 row, 2 cols, 2nd subplot

    x_EE = np.linspace(0, N*h, N+1)
    x_IM = np.linspace(0, N*h, N+1)
    ax2_0.semilogy(x_IM, np.abs(E1(sol_IMP)-E1(y0)), 'b-',label="Implicit Mid-Point")
    ax2_0.semilogy(x_IM, np.abs(E1(sol_EE)-E1(y0)), 'k-',label="Explicit Euler")
    ax2_0.set_title(r"Energy error of $I_1$",fontsize=20)
    ax2_0.set_xlabel(r'$t$',fontsize=20)
    ax2_0.set_ylabel(r'$|I_1(\mathbf{x}(t))-I_1(\mathbf{x}_0)|$',fontsize=20)
    ax2_0.tick_params(axis='x', labelsize=15)
    ax2_0.tick_params(axis='y', labelsize=15)
    ax2_0.yaxis.get_offset_text().set_fontsize(15)

    ax2_1 = fig.add_subplot(gs[1,1])  # 1 row, 2 cols, 2nd subplot

    x_EE = np.linspace(0, N*h, N+1)
    x_IM = np.linspace(0, N*h, N+1)
    ax2_1.semilogy(x_IM, np.abs(E2(sol_IMP)-E2(y0)), 'b-',label="Implicit Mid-Point")
    ax2_1.semilogy(x_IM, np.abs(E2(sol_EE)-E2(y0)), 'k-',label="Explicit Euler")
    ax2_1.set_title(r"Energy error of $I_2$",fontsize=20)
    ax2_1.set_xlabel(r'$t$',fontsize=20)
    ax2_1.set_ylabel(r'$|I_2(\mathbf{x}(t))-I_2(\mathbf{x}_0)|$',fontsize=20)
    ax2_1.tick_params(axis='x', labelsize=15)
    ax2_1.tick_params(axis='y', labelsize=15)
    ax2_1.yaxis.get_offset_text().set_fontsize(15)


def plot_energy_behaviour_SIR(soln,t_eval,method):

    E = np.sum(soln,axis=0)

    plt.plot(t_eval,soln[0],'r-',label=r"$S$")
    plt.plot(t_eval,soln[1],'k-',label=r"$I$")
    plt.plot(t_eval,soln[2],'b-',label=r"$R$")
    plt.plot(t_eval,E,'c-',label=r"$S+I+R$",linewidth=2)
    plt.xlabel(r"$t$")
    plt.legend()
    plt.show();

def get_plots_LotkaVolterra(soln,t_eval,method):

    plt.subplot(1, 2, 1)
    plt.plot(soln[0],soln[1],'k-')
    plt.plot(soln[0,0],soln[1,0],'ro',label="Initial condition")
    plt.plot(gamma/delta,alpha/beta,'bo',label="Equilibrium")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend(loc="center right")

    plt.subplot(1, 2, 2)
    E = [FI_LotkaVolterray(soln[:,i]) for i in range(len(t_eval))]
    E0 = FI_LotkaVolterray(soln[:,0])
    plt.semilogy(t_eval,np.abs(E0 - E),'k-',linewidth=0.5)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\left|E(x(t),y(t))-E(x_0,y_0)\right|$")

    plt.suptitle(f"Results with {method}")

    plt.show();