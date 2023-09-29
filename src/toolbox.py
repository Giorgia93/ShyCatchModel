# -*- coding: utf-8 -*-
"""
Toolbox module

"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd
from datetime import datetime


def calc_error(model, data):
    """
    Function that calculates the distance (error) from the model results to the 
    capture data

    Parameters
    ----------
    model : DataFrame
        Dataframe containing a column for each simulated date and corresponding
        number of daily captures
    data : DataFrame
        Dataframe containing a column for each date and corresponding data on 
        daily captures

    Returns
    -------
    error : float
        Error associated with given set of model results and data

    """
    # Merge model results and data by date
    merged_captures = pd.merge(model, data, how='inner', 
                               left_on='date', right_on='date').dropna()
    nb_observations = len(merged_captures)
    
    # Calculate residuals 
    residuals = merged_captures.data_captures - merged_captures.model_captures
    
    # Calculate mean sum of squares from residuals
    error = np.sum(residuals ** 2) * (1/nb_observations)
    
    return error


def choose_lure_change_scenario(all_scenarios, scenario_label):
    """
    Function returning the sequence of proportions of traps lured with lure A
    or lure B, before and after change of lure, for a given scenario label

    Parameters
    ----------
    all_scenarios : 2d float array
        Array containing the three possible combinations of proportions of lure
        A and lure B
    scenario_label : string
        Label of scenario wanted. Currently only three defined
        
    Returns
    -------
    lure_prop : 2d float array
        Array with proportions of traps lured with lure A and lure B, before 
        and after change of lure
        
    """
    
    if scenario_label == "A":
        lure_prop = [all_scenarios[0], all_scenarios[0]]
    
    elif scenario_label == "AB":
        lure_prop = [all_scenarios[1], all_scenarios[1]]
    
    elif scenario_label == "A->B":
        lure_prop = [all_scenarios[0], all_scenarios[2]]
    
    else:
        print("Scenario label not recognised.")
    
    return lure_prop
    

def create_results_struc(t0, nb_days, nb_captures):
    """
    Function that creates a dataframe of results with a column for dates and 
    one for number of daily captures from model results

    Parameters
    ----------
    t0 : datetime
        First day of simulation
    nb_days : int
        Number of simulated days
    nb_captures : float array
        Array of daily captures

    Returns
    -------
    results : DataFrame
        Dataframe of number of captures for each date modelled

    """
    dates = [t0+pd.DateOffset(i) for i in range(nb_days)]
    results = pd.DataFrame({'date' : dates, 'model_captures' : nb_captures})
    return results


def create_results_struc_jumpdate(t0, nb_days, nb_captures, data):
    """
    Function that creates a dataframe of results with a column for dates and 
    one for number of daily captures from model results

    Parameters
    ----------
    t0 : datetime
        First day of simulation
    nb_days : int
        Number of simulated days
    nb_captures : float array
        Array of daily captures

    Returns
    -------
    results : DataFrame
        Dataframe of number of captures for each date modelled

    """
    dates = [t0+pd.DateOffset(i) for i in range(nb_days)]
    model_results = pd.DataFrame({'date' : dates, 'model_captures' : nb_captures})
    
    # Merge model results and data by date
    merged_captures = pd.merge(model_results, data, how='inner', 
                               left_on='date', right_on='date')
    
    # Find indexes of dates where traps were checked
    date_indexes = np.append(-1, np.where(np.isnan(merged_captures.data_captures)==False))
    
    # Sum model captures for each date calculated above
    model_captures_jumpdate = np.array(\
        [np.sum(merged_captures.model_captures[date_indexes[i-1]+1:date_indexes[i]+1]) 
         for i in np.arange(1, len(date_indexes))])
        
    results = pd.DataFrame({'date' : data.date[np.isnan(data.data_captures)==False], 
                                               'model_captures' : model_captures_jumpdate})
        
    return results


def filter_trap_locations(trap_data, period):
    active_traps = np.array([trap_data.active_period1, 
                             trap_data.active_period2, 
                             trap_data.active_period3])
    active_traps = active_traps.astype(bool)
    active_trap_loc = np.array([trap_data.new_x[active_traps[period-1, :]], 
                                trap_data.new_y[active_traps[period-1, :]]])
    
    return np.transpose(active_trap_loc)


def filter_active_traps(trap_data, all_distances_from_traps, period):
    active_traps = np.array([trap_data.active_period1, 
                             trap_data.active_period2, 
                             trap_data.active_period3]).astype(bool)
    filtered_dist = np.array(all_distances_from_traps[active_traps[period-1, :], :])
    
    return filtered_dist
    
 
def get_all_p_encounter(hr_radii, distances, beta_mean, g0s_fit_coeff):
    """
    Function to generate all possible p_encounterTOT for any hr radius and any
    distance from traps

    Parameters
    ----------
    hr_radii : float 1d array
        Array of possible HR radii
    distances : float 2d array
        Array of distances between home-range centres and traps
    beta_mean : float
        Mean of original p_int beta distribution
    g0s_fit_coeff : 1x2 float array
        Fitted coefficients from the power law g0 = a*sigma^b

    Returns
    -------
    p_enc_tot : 2d array
        Nightly probability of encounter for each individual (cols), 
        for each hr size (rows)

    """
    
    # Sigma parameter of the half normal, calculated from given home-range size 
    # using the variance formula of the half-normal distribution
    sig = hr_radii / 2.45;
    
    # Probability of encounter at distance = 'dist', p_enc <= 1 (animal will 
    # cover entire home-range for small enough home-ranges and high enough 
    # diffusion coefficients
    p_enc = np.array([[((g0s_fit_coeff[0] * (s ** g0s_fit_coeff[1])) * 
              np.exp(-(d ** 2) / (2 * (s ** 2)))) / beta_mean for
               s in sig] for d in distances])
    # p_enc = ((g0s_fit_coeff[0] * (sig ** g0s_fit_coeff[1])) * 
    #           np.exp(-(distances ** 2) / (2 * (sig ** 2)))) / beta_mean 
    p_enc[p_enc > 1] = 1
    
    # Calculate total p_enc taking in consideration all traps within home 
    # range
    p_enc_tot = 1 - np.prod(1 - p_enc, axis=0)
    
    return p_enc_tot


def get_beta_params(mu, var):
    """
    Function that takes the mean and variance of a beta distribution and
    returns the corresponding alpha and beta parameters

    """
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2;
    beta = alpha * (1 / mu - 1);
    
    return alpha, beta


def get_distances_from_traps(trap_spacing, hr_radius, perception_dist, 
                              max_hrradius, Kcap):
    # Create an array of possible hr-centre positions in a square
    # with traps at each corner
    
    # Create matrix of all possible distances from traps
    distances_matrix = get_distances_matrix(trap_spacing, hr_radius, 
                                            perception_dist*2, max_hrradius)
    
    # Draw random hr centres/distances from distances_matrix
    rand_cols = np.random.randint(0, np.size(distances_matrix, 1) - 1, Kcap)
    distances = distances_matrix[:, rand_cols]
    
    ## Uncomment to see plot of traps and home ranges 
    # plot_traps_and_hr_locations(trap_loc, hrc_xy[rand_cols, :], hr_radius)
    
    return distances


def get_distances_from_trap_data(trap_loc, perception_dist, Kcap, hr_radius):
    """
    Takes a set of relative coordinates for each trap in a landscape, draws 
    random home range centres within the available space and calculates the 
    distance from each home-range centre to each trap

    Parameters
    ----------
    trap_loc : TYPE
        DESCRIPTION.
    perception_dist : TYPE
        DESCRIPTION.
    Kcap : TYPE
        DESCRIPTION.
    hr_radius : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    distances : TYPE
        DESCRIPTION.

    """
    # Create an array of possible hr-centre positions in a square
    # with traps at each corner
    
    # Load relative positions of traps as coordinates from file
    # plot_traps_locations(trap_loc)
    [hrc_xy, distances_matrix] = get_distances_matrix_from_data(trap_loc, perception_dist*2)

    # Draw random hr centres/distances from distances_matrix
    rand_cols = np.random.randint(0, np.size(distances_matrix, 1) - 1, Kcap)
    distances = distances_matrix[:, rand_cols]
    
    ## Uncomment to see plot of traps and home ranges 
    # plot_traps_and_hr_locations(trap_loc, hrc_xy[rand_cols, :], hr_radius)
    
    return hrc_xy[rand_cols], distances


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
    
    
def get_distances_matrix_from_data(trap_loc_data, cell_size):
    """
    Create a grid of cells (potential hr centre locations) and returns a matrix 
    of distances from each cell to all traps within maximum home-range size

    Parameters
    ----------
    trap_loc_data : table
        Table containing columns for the x and y coordinates of each trap, 
        projected so that the first trap from the west is at x=100m and the 
        first trap up from the south is at y=100
    cell_size : float
        Side of each square cell. Calculated as twice the perception distance
        
    Returns
    -------
    distances_matrix : 2D float array
        Matrix with euclidean distances from each cell to all traps within 
        hr_size. Each column of the distance matrix contains the distances from 
        a single cell to all traps.

    """
    
    # Get max hr centre coordinates (min is [0, 0])
    max_hrc = [max(trap_loc_data.new_x) + 100, 
               max(trap_loc_data.new_y) + 100]
        
    # Cells coordinates (= hr_centres coordinates)
    cell_x = np.arange(0, max_hrc[0], cell_size)
    cell_y = np.arange(0, max_hrc[1], cell_size)
    
    A, B = np.meshgrid(cell_x, cell_y)
    c = np.concatenate((np.transpose(A), np.transpose(B)), axis=1)
    cell_xy = np.reshape(c, (-1,2) , order='F')
    
    trap_xy = np.transpose(np.array([trap_loc_data.new_x, trap_loc_data.new_y]))
    
    # Matrix of distances cells/traps (each cell in a column)
    distances_matrix = cdist(trap_xy, cell_xy)  

    return cell_xy, distances_matrix


def get_home_range_radius(max_hr_radius, pop_density, sD_fit_coeff):
    """
    Function that gets animals' home range radius according to given 
    population densities, using log-log lm coefficients defined in params.py
    
    Parameters
     ----------
    max_hr_radius: float
        Maximum hr radius
    pop_density: float
        Population density
    fit_coeff : 1x2 float array
        Array containing the coefficients b0 and b1 of the model 
        log(sigma) = b0 + b1*log(density), as described in the Vattiato23 
        paper on pest detectability
    
    Returns
    -------
    hr_radius : float
        Updated home-range radius in meters
        
    """
    # Use log-log lm fit parameters from pest detectability data to updat sigma
    sigma = np.exp(sD_fit_coeff[0] + np.log(pop_density) * sD_fit_coeff[1])
    
    # Convert sigma to hr_size (circular area in m2)
    new_hr_area = np.pi * (2.45 * sigma) ** 2
    
    # Calculate hr radius (m) from hr area (m2)
    hr_radius = min(max_hr_radius, np.sqrt(new_hr_area / np.pi))
    
    return hr_radius


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
    while a <= 0:
        beta_var *= 0.9
        a, b = get_beta_params(beta_mean, beta_var)
    
    # Draw N number of random probabilities of interaction from distribution 
    p_int = np.random.beta(a, b, N)
    
    return p_int


def get_random_samples(pars_to_fit, n_samples):
    """
    Function that takes a dictionary of lower and upper bounds for each 
    parameter to estimate and returns an array of random samples drawn from
    a uniform distribution with given bounds.

    Parameters
    ----------
    pars_to_fit : dict
        Dictionary of the form {'par_to_sample1' : [lb, ub], ...}
    n_samples : int
        Number of random samples required

    Returns
    -------
    samples : float array
        Array of uniform random samples (each column is a parameter, each row
                                         is a sample)

    """
    rand_samples = np.zeros((n_samples, len(pars_to_fit)))
    for i, par in enumerate(pars_to_fit.keys()):
        rand_samples[:, i] = np.random.uniform(pars_to_fit[par][0], 
                                         pars_to_fit[par][1], n_samples)
        
    return rand_samples
    

def import_data(filename):
    """
    Function to import capture data from specified folder and csv file

    Parameters
    ----------
    folder_name : str
        Folder where data is stored, e.g. "../data/"
    filename : str
        Filename of csv file to import, e.g. "capture_data.csv"

    Returns
    -------
    df : DataFrame
        Dataframe including dates and capture data

    """
    # Create data frame from csv with pandas module
    df = pd.read_csv(filename, 
                     names=['date', 'data_captures'], 
                     parse_dates=[0], date_format="dd-mmm", skiprows=1)
    df.date = [datetime.strptime(date, "%d-%b") for date in df.date]
    return df


def moving_average(array, window_size, mov_type=1):
    
    """
    Function to calculate the rolling average of a given array or matrix over a 
    set window. If a 2d array is given as input, the rolling average is 
    calculated for each line of the array independently of the others.

    Parameters
    ----------
    array : float array
        Array of values to average
    window_size : int
        Number of elements over which to average

    Returns
    -------
    moving_averages : float array
        Array of rolling averages

    """
    nb_trajectories = np.size(array, 0)
    nb_timepoints = np.size(array, 1)
    moving_averages = np.empty((nb_trajectories, nb_timepoints))
    
    for i in range(nb_timepoints):
        if mov_type == 0:
            # Store elements from i-window_size/2 to i+window_size/2 in list to  
            # get the current window for each trajectory
            window = array[:, max(0, i-int(window_size/2)): 
                           min(len(array), i+int(window_size/2))]
            
        elif mov_type == 1:
            # Store elements from i-window_size to i in list to get the current 
            # window for each trajectory
            window = array[:, max(0, i-window_size):i]
    
        # Calculate the average of current window
        window_average = np.sum(window, axis=1) / np.size(window, 1)
        
        # Store the average of current window in moving average list
        moving_averages[:, i] = window_average
            
        
    return moving_averages


def plot_beta_distribution(mean, var):
    a, b = get_beta_params(mean, var)
    x = np.arange(0, 1, 0.01)
    y = beta.pdf(x, a, b)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel("$p_{int}$")
    ax.set_ylabel("beta PDF")


def plot_captures_comparison(model_t0, model_nb_days, model_captures, 
                             model_captures_jumpdate, model_N, model_penc, 
                             model_hr, model_pint, data_captures, errors):
    nb_traj = min(20, len(model_captures)-1)        # Number of model trajectories to plot
    mav_window = 2     # (days) moving average window for model results
    error_norm = 1 - (errors-np.min(errors)) / (np.max(errors)-np.min(errors))
    
    # Crop model at same dates as data
    model_t = [model_t0 + pd.DateOffset(i) for i in range(len(data_captures))]
    model_c = moving_average(model_captures[:, 0:len(data_captures)], mav_window)
    model_c2 = moving_average(model_captures_jumpdate[:, 0:len(data_captures)], mav_window)
    model_n = model_N[:, 0:len(data_captures)]
    model_pe = model_penc[:, 0:len(data_captures)]
    model_hr = model_hr[:, 0:len(data_captures)]
    model_pi = model_pint[:, 0:len(data_captures)]
    model_cb = np.array([np.min(model_c, axis=0), np.max(model_c, axis=0)])
    model_c2b = np.array([np.min(model_c2, axis=0), np.max(model_c2, axis=0)])
    model_nb = np.array([np.min(model_n, axis=0), np.max(model_n, axis=0)])
    model_peb = np.array([np.min(model_pe, axis=0), np.max(model_pe, axis=0)])
    model_hrb = np.array([np.min(model_hr, axis=0), np.max(model_hr, axis=0)])
    model_pib = np.array([np.min(model_pi, axis=0), np.max(model_pi, axis=0)])
    
    data_t_filt = data_captures.date[np.isnan(data_captures.data_captures) == False]
    
    ifig = 0
    
    fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(8, 20))
    
    # Plot daily capture history
    axs[ifig].fill_between(data_t_filt, model_c2b[0, :], 
                        model_c2b[1, :], color="lightgray")
    for i in range(nb_traj):
        axs[ifig].plot(data_t_filt, model_c2[i, :], color="black", alpha = error_norm[i])
    axs[ifig].plot(data_t_filt, model_c2[nb_traj, :], color="black", alpha = error_norm[nb_traj], label="model")
    axs[ifig].plot(data_captures.date, data_captures.data_captures, 'rx', 
            label="data")
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ifig].set_ylabel("captures")
    axs[ifig].set_ylim(top=70)
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    axs[ifig].legend(loc="upper right")
    
    ifig += 1
    
    # Plot daily capture history
    axs[ifig].fill_between(model_t[0:len(data_captures)], model_cb[0, :], 
                        model_cb[1, :], color="lightgray")
    for i in range(nb_traj):
        axs[ifig].plot(model_t, model_c[i, :], color="black", alpha = error_norm[i])
    axs[ifig].plot(model_t, model_c[nb_traj, :], color="black", alpha = error_norm[nb_traj])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ifig].set_ylabel("daily captures")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    axs[ifig].legend(loc="upper right")
    
    ifig += 1
    
    # Plot population size history
    axs[ifig].fill_between(model_t[0:len(data_captures)], model_nb[0, :], 
                        model_nb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_n[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ifig].set_ylabel("population size")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    # Plot mean p_enc history
    axs[ifig].fill_between(model_t[0:len(data_captures)], model_peb[0, :], 
                        model_peb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_pe[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("mean $p_{encTOT}$")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    # Plot mean home-range radius history
    axs[ifig].fill_between(model_t[0:len(data_captures)], model_hrb[0, :], 
                        model_hrb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_hr[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("HR radius (m)")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    # Plot mean p_int history
    axs[ifig].fill_between(model_t[0:len(data_captures)], model_pib[0, :], 
                        model_pib[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_pi[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("mean $p_{int}$")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    

def plot_single_scenario(model_t0, model_nb_days, model_captures, model_N, 
                             model_penc, model_hr, model_pint, errors):
    nb_traj = min(20, len(model_captures)-1)        # Number of model trajectories to plot
    mav_window = 7     # (days) moving average window for model results
    error_norm = 1 - (errors-np.min(errors)) / (np.max(errors)-np.min(errors))
    
    model_t = [model_t0 + pd.DateOffset(i) for i in range(len(model_captures[0]))]
    model_c = moving_average(model_captures, mav_window)
    model_cb = np.array([np.min(model_c, axis=0), np.max(model_c, axis=0)])
    model_nb = np.array([np.min(model_N, axis=0), np.max(model_N, axis=0)])
    model_peb = np.array([np.min(model_penc, axis=0), np.max(model_penc, axis=0)])
    model_hrb = np.array([np.min(model_hr, axis=0), np.max(model_hr, axis=0)])
    model_pib = np.array([np.min(model_pint, axis=0), np.max(model_pint, axis=0)])
    
    ifig = 0
    
    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(15, 15))
    
    # Plot daily capture history
    axs[ifig].fill_between(model_t, model_cb[0, :], 
                        model_cb[1, :], color="lightgray")
    for i in range(nb_traj):
        axs[ifig].plot(model_t, model_c[i, :], color="black", alpha = error_norm[i])
    axs[ifig].plot(model_t, model_c[nb_traj, :], color="black", alpha = error_norm[nb_traj])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ifig].set_ylabel("daily captures")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    axs[ifig].legend(loc="upper right")
    
    ifig += 1
    
    # Plot population size history
    axs[ifig].fill_between(model_t, model_nb[0, :], 
                        model_nb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_N[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ifig].set_ylabel("population size")
    axs[ifig].set_xlabel("date")
    axs[ifig].set_ylim(bottom=0)
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    
    # Plot mean p_enc history
    axs[ifig].fill_between(model_t, model_peb[0, :], 
                        model_peb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_penc[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("mean $p_{encTOT}$")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    # Plot mean home-range radius history
    axs[ifig].fill_between(model_t, model_hrb[0, :], 
                        model_hrb[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_hr[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("HR radius (m)")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    
    ifig += 1
    
    # Plot mean p_int history
    axs[ifig].fill_between(model_t, model_pib[0, :], 
                        model_pib[1, :], color="lightgray")
    for i in range(nb_traj+1):
        axs[ifig].plot(model_t, model_pint[i, :], color="black", alpha = error_norm[i])
    axs[ifig].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[ifig].set_ylabel("mean $p_{int}$")
    axs[ifig].set_xlabel("date")
    axs[ifig].grid(alpha=0.5)
    

def plot_scenarios_heatmaps(all_results, scenarios):
    """
    Code to plot a comparison of scenario outputs as a heatmap

    Parameters
    ----------
    all_results : 3D float array
        Array of results to compare. Rows: days; Columns: output; Sheets: scenarios

    Returns
    -------
    None.

    """
    final_pop_med = np.median(all_results[:, 2, :], axis=0).reshape((2, 2)).astype(int)
    final_pop_min = np.quantile(all_results[:, 2, :], q=0.025, axis=0).reshape((2, 2)).astype(int)
    final_pop_max = np.quantile(all_results[:, 2, :], q=0.975, axis=0).reshape((2, 2)).astype(int)
    final_pint_med = np.nanmedian(all_results[:, 4, :], axis=0).reshape((2, 2))
    final_pint_min = np.nanquantile(all_results[:, 4, :], q=0.025, axis=0).reshape((2, 2))
    final_pint_max = np.nanquantile(all_results[:, 4, :], q=0.975, axis=0).reshape((2, 2))
    
    thresholds_labels = ["80%", "95%"]
    new_eff_labels = ["$p_{int}$ x3", "$p_{int}$ x2"]
    
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
    ifig = 0
    axs[ifig].imshow(final_pop_med, cmap='Reds')
    axs[ifig].set_xlabel("Threshold to change lure\n(% drop in captures compared to 1st week )")
    axs[ifig].set_xticks(np.arange(len(thresholds_labels)), labels=thresholds_labels)

    axs[ifig].set_ylabel("New lure effectiveness")
    axs[ifig].set_yticks(np.arange(len(thresholds_labels)), labels=new_eff_labels)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(thresholds_labels)):
        for j in range(len(new_eff_labels)):
            lab = str(final_pop_med[i, j]) + "\n[" + str(final_pop_min[i, j]) + \
                " - " + str(final_pop_max[i, j]) + "]"
            axs[ifig].text(j, i, lab, ha="center", va="center", color="k")
            
    axs[ifig].set_title("Surviving population after 8 months of trapping\n($N_0$=171)")
            
    ifig += 1
    axs[ifig].imshow(final_pint_med, cmap='Reds')
    axs[ifig].set_xlabel("Threshold to change lure\n(% drop in captures compared to 1st week)")
    axs[ifig].set_xticks(np.arange(len(thresholds_labels)), labels=thresholds_labels)

    axs[ifig].set_ylabel("New lure effectiveness")
    axs[ifig].set_yticks(np.arange(len(thresholds_labels)), labels=new_eff_labels)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(thresholds_labels)):
        for j in range(len(new_eff_labels)):
            lab = str(round(final_pint_med[i, j],3)) + "\n[" + \
                str(round(final_pint_min[i, j],3)) + \
                " - " + str(round(final_pint_max[i, j],3)) + "]"
            axs[ifig].text(j, i, lab, ha="center", va="center", color="k")
            
    axs[ifig].set_title("Avg. $p_{int}$ of surviving population after 8 months of trapping")
      
    
def plot_timeseries(t, N_history, penc_history, pint_history, 
                   mean_penc_history, mean_pint_history, 
                   mean_pc_history, hr_radius_history, alive_history, 
                   beta_mean, beta_var, change_lure_day):
    
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
    axs[1].plot(range(t), N_history[0:t])
    axs[1].axvline(x=change_lure_day, color="red", linestyle="--", 
                   label="Change of lure")
    axs[1].set_ylabel("Population size")
    axs[1].set_xlabel("time (days)")
    axs[1].grid(alpha=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlim(right=1000)
    if not np.isnan(change_lure_day):
        axs[1].legend(loc="upper right")
    
    # Plot mean p_enc and home-range radius on same plot
    axs[2].plot(range(t), mean_penc_history[0:t])
    axs[2].set_ylabel("mean $p_{encTOT}$", color="#1f77b4")
    axs[2].set_xlabel("time (days)")
    axs[2].tick_params(axis='y', colors="#1f77b4")
    axs[2].spines['left'].set_color('#1f77b4')
    axs[2].grid(alpha=0.5)
    axs[2].set_axisbelow(True)
    axs[2].set_xlim(right=1000)
    
    ax2 = axs[2].twinx()
    ax2.plot(range(t), hr_radius_history[0:t], color=color)
    ax2.set_ylabel("HR radius (m)", color=color)
    ax2.set_xlabel("time (days)")
    ax2.tick_params(axis='y', colors=color)
    ax2.spines['right'].set_color(color)
    
    # Plot mean p_int 
    axs[3].plot(range(t), mean_pint_history[0:t])
    axs[3].axvline(x=change_lure_day, color="red", linestyle="--", 
                   label="Change of lure")
    axs[3].set_ylabel("mean $p_{int}$")
    axs[3].set_xlabel("time (days)")
    axs[3].grid(alpha=0.5)
    axs[3].set_axisbelow(True)
    axs[3].set_xlim(right=1000)
    if not np.isnan(change_lure_day):
        axs[3].legend(loc="upper right")
    
    fig.tight_layout()
    plt.show()
    
    
def plot_traps_locations(location_data):
    
    x1 = [x for i, x in enumerate(location_data.new_x) if 
          location_data.active_period1[i]==1]
    y1 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==1]
    x2 = [x for i, x in enumerate(location_data.new_x) if\
          location_data.active_period1[i]==0 and location_data.active_period2[i]==1]
    y2 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==0 and location_data.active_period2[i]==1]
    x3 = [x for i, x in enumerate(location_data.new_x) if
          location_data.active_period1[i]==0 and 
          location_data.active_period2[i]==0 and location_data.active_period3[i]==1]
    y3 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==0 and 
          location_data.active_period2[i]==0 and location_data.active_period3[i]==1]
    
    minor_ticks_x = np.arange(0, max(location_data.new_x) + 100, 100)
    minor_ticks_y = np.arange(0, max(location_data.new_y) + 100, 100)
    
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.scatter(x1, y1, color='r', marker='x', linewidth=3, label="Active from date 1")
    ax.scatter(x2, y2, color='b', marker='x', linewidth=3, label="Active from date 2")
    ax.scatter(x3, y3, color='g', marker='x', linewidth=3, label="Active from date 3")
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim((0, max(location_data.new_x) + 100))
    ax.set_ylim((0, max(location_data.new_y) + 100))
    ax.set_xlabel("(m)")
    ax.set_ylabel("(m)")
    

def plot_traps_and_hr_locations(location_data, hrc_coord, hr_radius):
    
    x1 = [x for i, x in enumerate(location_data.new_x) if 
          location_data.active_period1[i]==1]
    y1 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==1]
    x2 = [x for i, x in enumerate(location_data.new_x) if\
          location_data.active_period1[i]==0 and location_data.active_period2[i]==1]
    y2 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==0 and location_data.active_period2[i]==1]
    x3 = [x for i, x in enumerate(location_data.new_x) if
          location_data.active_period1[i]==0 and 
          location_data.active_period2[i]==0 and location_data.active_period3[i]==1]
    y3 = [x for i, x in enumerate(location_data.new_y) if
          location_data.active_period1[i]==0 and 
          location_data.active_period2[i]==0 and location_data.active_period3[i]==1]
    
    minor_ticks_x = np.arange(0, max(location_data.new_x) + 100, 100)
    minor_ticks_y = np.arange(0, max(location_data.new_y) + 100, 100)
    
    fig, ax = plt.subplots(figsize=(10, 15))
    for hrc in hrc_coord:
        circle = plt.Circle((hrc), hr_radius, color='lightgray', fill=False)
        ax.add_patch(circle)
    ax.scatter(x1, y1, color='r', marker='x', linewidth=3, label="Active from date 1")
    ax.scatter(x2, y2, color='b', marker='x', linewidth=3, label="Active from date 2")
    ax.scatter(x3, y3, color='g', marker='x', linewidth=3, label="Active from date 3")
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim((0, max(location_data.new_x) + 100))
    ax.set_ylim((0, max(location_data.new_y) + 100))
    ax.set_xlabel("(m)")
    ax.set_ylabel("(m)")
    

def plot_pint_dist(t, pint_history, alive_history):
    
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(4, 5))
    
    # Plot histogram of previous personality distribution
    axs[0].hist(x=pint_history[alive_history[:, t-1] == True, t-1], 
                density=True, bins=20, edgecolor='black')
    axs[0].set_title("$p_{int}$ distribution at $t_{prev}$")
    axs[0].grid(alpha=0.5)
    axs[0].set_axisbelow(True)
    axs[0].set_ylabel("PDF")
    
    # Plot histogram of current personality distribution
    axs[1].hist(x=pint_history[alive_history[:, t] == True, t], 
                density=True, bins=20, edgecolor='black')
    axs[1].set_title("$p_{int}$ distribution at $t_{curr}$")
    axs[1].grid(alpha=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_ylabel("PDF")
    axs[1].set_xlim(left=0)
    
    fig.tight_layout()
    plt.show()
    

def plot_posteriors(posterior, priors):
    
    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(11, 9))
    title_ypos = 1.01
    
    var = posterior.get("beta_mean")
    prior_bounds = priors.get("beta_mean")
    best_mean = var[np.argmin(posterior.error)]
    axs[0, 0].hist(x = var, bins=20, weights = 100*np.ones_like(var)/len(var),
                edgecolor='black', color='grey', label="posterior")
    axs[0, 0].axvline(x = best_mean, color='r', linewidth=2)
    axs[0, 0].grid(alpha=0.5)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_xlabel("Initial mean $\mu_0$ of population's $p_{int}$")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].yaxis.set_major_formatter(PercentFormatter())
    axs[0, 0].set_xlim([prior_bounds[0], prior_bounds[1]])
    axs[0, 0].set_title("(a)", loc='left', weight='bold', y=title_ypos)
    
    var = posterior.get("beta_var")
    prior_bounds = priors.get("beta_var")
    best_var = var[np.argmin(posterior.error)]
    axs[0, 1].hist(x = var, bins=20, weights = 100*np.ones_like(var)/len(var),
                edgecolor='black', color='grey', label="posterior")
    axs[0, 1].axvline(x = best_var, color='r', linewidth=2)
    axs[0, 1].grid(alpha=0.5)
    axs[0, 1].set_axisbelow(True)
    axs[0, 1].set_xlabel("Initial variance $\sigma_0^2$ of population's $p_{int}$")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].yaxis.set_major_formatter(PercentFormatter())
    axs[0, 1].set_xlim([prior_bounds[0], prior_bounds[1]])
    axs[0, 1].set_title("(b)", loc='left', weight='bold', y=title_ypos)
    
    var = posterior.get("N0")
    prior_bounds = priors.get("N0")
    axs[1, 0].hist(x = var, bins=20, weights = 100*np.ones_like(var)/len(var),
                edgecolor='black', color='grey', label="posterior")
    axs[1, 0].axvline(x = var[np.argmin(posterior.error)], color='r', linewidth=2)
    axs[1, 0].grid(alpha=0.5)
    axs[1, 0].set_axisbelow(True)
    axs[1, 0].set_xlabel("Initial population size $N_0$")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].yaxis.set_major_formatter(PercentFormatter())
    axs[1, 0].set_xlim([prior_bounds[0], prior_bounds[1]])
    axs[1, 0].set_title("(c)", loc='left', weight='bold', y=title_ypos)
    
    (a, b) = get_beta_params(best_mean, best_var)
    x = np.linspace(0, 1, 1000)
    axs[1, 1].plot(x, beta.pdf(x, a, b), 'r-', lw=2)
    axs[1, 1].set_xlabel("Probability of interaction $p_{int}$")
    axs[1, 1].set_ylabel("PDF")
    axs[1, 1].set_title("(d)", loc='left', weight='bold', y=title_ypos)
        
    # fig.suptitle('Posterior distributions')
    fig.tight_layout()
    plt.show()
    

def produce_summary_tables(all_results, scenarios):
    """
    ...

    Parameters
    ----------
    all_results : 3D float array
        Array of results to compare. Rows: days; Columns: output; Sheets: scenarios

    Returns
    -------
    None.

    """
    nscen = len(scenarios)
    final_pop_med = np.median(all_results[:, 2, :], axis=0).astype(int)
    final_pop_min = np.quantile(all_results[:, 2, :], q=0.025, axis=0).astype(int)
    final_pop_max = np.quantile(all_results[:, 2, :], q=0.975, axis=0).astype(int)
    final_pint_med = np.nanmedian(all_results[:, 4, :], axis=0)
    final_pint_min = np.nanquantile(all_results[:, 4, :], q=0.025, axis=0)
    final_pint_max = np.nanquantile(all_results[:, 4, :], q=0.975, axis=0)
    
    # Repetition-wise differences, use if simulations were run using different
    # parameter sets
    final_pop_diff = 100 * (all_results[:, 2, 1:] - all_results[:, 2, np.repeat(0, nscen-1)]) / all_results[:, 2, np.repeat(0, nscen-1)]
    final_pop_diff_med = np.median(final_pop_diff, axis=0).astype(int)
    final_pop_diff_min = np.quantile(final_pop_diff, q=0.025, axis=0).astype(int)
    final_pop_diff_max = np.quantile(final_pop_diff, q=0.975, axis=0).astype(int)
    final_pint_diff = 100 * (all_results[:, 4, 1:] - all_results[:, 4, np.repeat(0, nscen-1)]) / all_results[:, 4, np.repeat(0, nscen-1)]
    final_pint_diff_med = np.median(final_pint_diff, axis=0).astype(int)
    final_pint_diff_min = np.quantile(final_pint_diff, q=0.025, axis=0).astype(int)
    final_pint_diff_max = np.quantile(final_pint_diff, q=0.975, axis=0).astype(int)
    
    # Overall differences, use if simulations were run using same param set
    final_pop_diff_min = 100 * (np.quantile(all_results[:, 2, 1:], q=0.025, axis=0) - 
                                np.quantile(all_results[:, 2, np.repeat(0, nscen-1)], q=0.975, axis=0)) / \
                                np.quantile(all_results[:, 2, np.repeat(0, nscen-1)], q=0.975, axis=0)
    final_pop_diff_max = 100 * (np.quantile(all_results[:, 2, 1:], q=0.975, axis=0) - 
                                np.quantile(all_results[:, 2, np.repeat(0, nscen-1)], q=0.025, axis=0)) / \
                                np.quantile(all_results[:, 2, np.repeat(0, nscen-1)], q=0.025, axis=0)
    final_pint_diff_min = 100 * (np.nanquantile(all_results[:, 4, 1:], q=0.025, axis=0) - 
                                np.nanquantile(all_results[:, 4, np.repeat(0, nscen-1)], q=0.975, axis=0)) / \
                                np.nanquantile(all_results[:, 4, np.repeat(0, nscen-1)], q=0.975, axis=0)
    final_pint_diff_max = 100 * (np.nanquantile(all_results[:, 4, 1:], q=0.975, axis=0) - 
                                np.nanquantile(all_results[:, 4, np.repeat(0, nscen-1)], q=0.025, axis=0)) / \
                                np.nanquantile(all_results[:, 4, np.repeat(0, nscen-1)], q=0.025, axis=0)
    
    print(u"Scenario\tFinal population\tMean p_int of final population")
    for iScen in range(len(scenarios)):
        print("{0}\t{1} [{2}, {3}]\t{4:.4f} [{5:.4f}, {6:.4f}]".format(scenarios.lure_prop_scenario[iScen], 
                                                         final_pop_med[iScen],
                                                         final_pop_min[iScen],
                                                         final_pop_max[iScen],
                                                         final_pint_med[iScen],
                                                         final_pint_min[iScen],
                                                         final_pint_max[iScen]))
        if iScen > 0:
            print("{0}\t{1:+g}% [{2:+.0f}%, {3:+.0f}%]\t{4:+g}% [{5:+.0f}%, {6:+.0f}%]".format("", 
                                                             final_pop_diff_med[iScen-1],
                                                             final_pop_diff_min[iScen-1],
                                                             final_pop_diff_max[iScen-1],
                                                             final_pint_diff_med[iScen-1],
                                                             final_pint_diff_min[iScen-1],
                                                             final_pint_diff_max[iScen-1]))
    
   