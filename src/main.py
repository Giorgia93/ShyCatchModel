"""
Created on Thu Jul 20 14:55:39 2023
@author: VattiatoGi

SHYCATCH Model
Used to simulate dynamics of wild populations of possums during an eradication 
program. Model includes adaptive home-range sizes, population dynamics, heterogeneous levels of trap-shyness, and change of lure.

"""
from parameters import Params
from population import Population
import numpy as np
import toolbox as tb
from copy import copy


def main():
    print("Hiiiiiyaaaaa")
    params = Params()
    pop = Population(params)
    
    N_history = np.zeros(params.t_max)  # Initialisation of density data array
    hr_radius_history = np.zeros(params.t_max)  # Initialisation of hr_radius data array
    penc_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_enc data matrix
    pint_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_enc data matrix
    pc_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_capture data matrix
    alive_history = np.ones((params.Kcap, params.t_max)) # Initialisation of alive/dead bool data matrix
    mean_penc_history = np.zeros(params.t_max)   # Initialisation of p_enc data matrix
    mean_pint_history = np.zeros(params.t_max)   # Initialisation of p_int data matrix
    mean_pc_history = np.zeros(params.t_max) # Initialisation of p_capture data matrix
    change_lure_day = float("nan")

    for t in range(params.t_max):
        
        if np.mod(t, 100) == 0:
            print("Day {0}. Population size = {1}".format(t, pop.pop_size))
        
        # If less than min_pop or time > 1000 days, stop simulation
        if pop.pop_size < params.min_pop:
            print('Eradication complete in {0} days. {1} possums left.'.format(
                t, pop.pop_size))
            break
        
        # Update population
        pop.update_population(t, params)
        
        # Update time series of mean p_enc, density, p_int, p_capt for plots
        N_history[t] = pop.pop_size
        hr_radius_history[t] = pop.hr_radius
        penc_history[:, t] = pop.p_enc
        pc_history[:, t] = pop.p_capture
        pint_history[:, t] = pop.p_int
        alive_history[:, t] = pop.alive
        mean_penc_history[t] = np.mean(pop.p_enc[pop.alive == 1])
        mean_pint_history[t] = np.mean(pop.p_int[pop.alive == 1])
        mean_pc_history[t] = np.mean(pop.p_capture[pop.alive == 1])
        
        # Change lure if rate of capture falls below a set value
        avg_daily_capture = np.abs(N_history[t-7] - pop.pop_size) / 7
        if params.new_lure_effectiveness > 0 and np.isnan(change_lure_day) and t >= 10 and avg_daily_capture < params.change_lure_threshold:
            pop.change_lure(params.new_lure_effectiveness)
            pint_history[:, t] = pop.p_int
            change_lure_day = copy(t)
            # tb.plot_pint_dist(t, pint_history, alive_history)
        
    # Plot timeseries at the end of simulation
    tb.plot_timeseries(t, N_history, penc_history, pint_history, 
                       mean_penc_history, mean_pint_history, 
                       mean_pc_history, hr_radius_history, 
                       alive_history, params.beta_mean, params.beta_var, 
                       change_lure_day)
    
    if t + 1 == params.t_max:
        print('Time s up, eradication not complete, {0} possums left after {1} days.'.format(
            pop.pop_size, params.t_max))

if __name__ == "__main__":
    main()