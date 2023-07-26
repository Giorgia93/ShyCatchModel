# -*- coding: utf-8 -*-
"""
Toolbox module

"""

import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt


def get_distances_from_traps(trap_spacing, hr_radius, perception_dist, 
                             max_hrradius, Kcap):
    # Create an array of possible hr-centre positions in a square
    # with traps at each corner
    distances_matrix = get_distances_matrix(trap_spacing, hr_radius, 
                                            perception_dist*2, max_hrradius)

    # Draw random hr centres/distances from distances_matrix
    rand_cols = np.random.randint(0, np.size(distances_matrix, 1) - 1, Kcap)
    distances = distances_matrix[:, rand_cols]
    
    return distances


def get_distances_matrix(trap_grid_spacing, hr_radius, cell_size, max_hrradius):
    """
    Create a grid of cells (potential hr centre locations) and returns a matrix 
    of distances from each cell to all traps within maximum home-range size

    Parameters
    ----------
    trap_grid_spacing : float
        Distance between traps in meters. (Assumes they are placed in a square 
        grid)
    hr_radius : float
        Radius of home range in meters.
    cell_size : float
        Side of each square cell. Calculated as twice the perception distance
    max_hrradius : float
        Maximum home range radius in meters.

    Returns
    -------
    distances_matrix : 2D float array
        Matrix with euclidean distances from each cell to all traps within 
        hr_size. Each column of the distance matrix contains the distances from 
        a single cell to all traps.

    """
    
    if trap_grid_spacing < cell_size:
        print("Error: trap spacing smaller than cell size")
        
    else:
        # Calculate number of traps on the side of the square grid of traps
        nb_traps_side = int(2 * (np.floor(hr_radius / trap_grid_spacing) + 1))
        
        # Calculate number of cells in the side of a square with traps at vertices
        nb_cells_side = int(np.ceil(trap_grid_spacing / cell_size) + 1)
        
        # Trap coord vectors
        trap_x = np.linspace(0, trap_grid_spacing * (nb_traps_side - 1), 
                             nb_traps_side)
        trap_y = np.linspace(0, trap_grid_spacing * (nb_traps_side - 1), 
                             nb_traps_side)
        A, B = np.meshgrid(trap_x, trap_y)
        c = np.concatenate((np.transpose(A), np.transpose(B)), axis=1)
        trap_xy = np.reshape(c, (-1,2) , order='F')
        
        # Add traps on either side of the square (to account for when hr_size 
        # increases)
        to_add = int(np.ceil(0.5 * (2 * (np.floor(max_hrradius / trap_grid_spacing) + 
                                  1 - nb_traps_side))))
        trap_x = np.concatenate([np.linspace(-to_add * trap_grid_spacing, 
                                             -trap_grid_spacing, 
                                             to_add), 
                                 trap_x, 
                                 np.linspace(trap_x[-1] + trap_grid_spacing, 
                                             trap_x[-1] + to_add * trap_grid_spacing, 
                                             to_add)])
        trap_y = np.concatenate([np.linspace(-to_add * trap_grid_spacing, 
                                             -trap_grid_spacing, 
                                             to_add),
                                 trap_y, 
                                 np.linspace(trap_y[-1] + trap_grid_spacing, 
                                             trap_y[-1] + to_add * trap_grid_spacing, 
                                             to_add)])
        
        # Cells coordinates (= hr_centres coordinates)
        min_hrc = int(trap_grid_spacing * (nb_traps_side / 2 - 1))
        max_hrc = int(trap_grid_spacing * (nb_traps_side / 2))
        cell_x = np.linspace(min_hrc, max_hrc, int(nb_cells_side))
        cell_y = np.linspace(min_hrc, max_hrc, int(nb_cells_side))
        A, B = np.meshgrid(cell_x, cell_y)
        c = np.concatenate((np.transpose(A), np.transpose(B)), axis=1)
        cell_xy = np.reshape(c, (-1,2) , order='F')
        
        
        # Matrix of distances cells/traps (each cell in a column)
        distances_matrix = cdist(trap_xy, cell_xy)  
    
        return distances_matrix
    
    
def get_beta_params(mu, var):
    """
    Function that takes the mean and variance of a beta distribution and
    returns the corresponding alpha and beta parameters

    """
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2;
    beta = alpha * (1 / mu - 1);
    
    return alpha, beta


def get_p_interaction(N, beta_mean, beta_var):
    """
    Function that converts a mean and variance of p_int into an array of 
    randomly drawn p_int

    Parameters
    ----------
    N : int
        Number of random p_int required
    beta_mean : float
        Mean of p_int beta distribution
    beta_var : float
        Variance of p_int beta distribution

    Returns
    -------
    p_int : float array
        Numpy array of random p_int drawn from specified distirbution

    """
    # Convert trappability distribution mean and variance into beta 
    # distribution parameters
    a, b = get_beta_params(beta_mean, beta_var)
    
    # Draw N number of random probabilities of interaction from distribution 
    p_int = np.random.beta(a, b, N)
    
    return p_int


def plot_timeseries(t, N_history, penc_history, pint_history, 
                   mean_penc_history, mean_pint_history, 
                   mean_pc_history, hr_radius_history, alive_history, 
                   beta_mean, beta_var):
    
    plt.close()
    
    color = 'tab:red'
    
    alive = alive_history[:, 0]
    
    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(4, 10))
    
    # Plot histogram of initial personality distribution
    axs[0].hist(x=pint_history[alive == True, 0], density=True, bins=20, 
                   edgecolor='black')
    axs[0].set_title("$p_{int}$ distribution at $t=0$")
    axs[0].grid(alpha=0.5)
    axs[0].set_axisbelow(True)
    axs[0].set_ylabel("PDF")
    axs[0].text(0.75, 6, '$\mu$ ={0:.1f}'.format(beta_mean))
    axs[0].text(0.75, 5, '$\sigma^2$={0:.1f}'.format(beta_var))
    
    # Plot population size history
    axs[1].plot(range(t+1), N_history[0:t+1])
    axs[1].set_ylabel("Population size")
    axs[1].set_xlabel("time (days)")
    axs[1].grid(alpha=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_ylim(bottom=0)
    
    # Plot mean p_enc and home-range radius on same plot
    axs[2].plot(range(t+1), mean_penc_history[0:t+1])
    axs[2].set_ylabel("mean $p_{encTOT}$", color="#1f77b4")
    axs[2].set_xlabel("time (days)")
    axs[2].tick_params(axis='y', colors="#1f77b4")
    axs[2].spines['left'].set_color('#1f77b4')
    axs[2].grid(alpha=0.5)
    axs[2].set_axisbelow(True)
    
    ax2 = axs[2].twinx()
    ax2.plot(range(t+1), hr_radius_history[0:t+1], color=color)
    ax2.set_ylabel("HR radius (m)", color=color)
    ax2.set_xlabel("time (days)")
    ax2.tick_params(axis='y', colors=color)
    ax2.spines['right'].set_color(color)
    
    # Plot mean p_int and mean p_capture on same plot
    axs[3].plot(range(t+1), mean_pint_history[0:t+1])
    axs[3].set_ylabel("mean $p_{int}$")
    axs[3].set_xlabel("time (days)")
    axs[3].grid(alpha=0.5)
    axs[3].set_axisbelow(True)
    
    fig.tight_layout()
    plt.show()
    # fig.delaxes(axs[0,1])
    
    
    
trap_grid_spacing = 200.
hr_radius = 150.
cell_size = 20.
max_hrradius = 400.
res = get_distances_matrix(200., 150., 20., 400.)