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


def main():
    print("Hiiiiiyaaaaa")
    params = Params()
    pop = Population(params)
    pop.print_pop_size()
    pop.print_hr_radius()
    
    N_history = np.zeros(params.t_max)  # Initialisation of density data array
    hr_radius_history = np.zeros(params.t_max)  # Initialisation of hr_radius data array
    penc_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_enc data matrix
    pint_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_enc data matrix
    pc_history = np.zeros((params.Kcap, params.t_max)) # Initialisation of p_capture data matrix
    alive_history = np.ones((params.Kcap, params.t_max)) # Initialisation of alive/dead bool data matrix
    mean_penc_history = np.zeros(params.t_max)   # Initialisation of p_enc data matrix
    mean_pint_history = np.zeros(params.t_max)   # Initialisation of p_int data matrix
    mean_pc_history = np.zeros(params.t_max) # Initialisation of p_capture data matrix


    for t in range(params.t_max):
        
        print("Day {0}. Population size = {1}".format(t, pop.pop_size))
        
        # Live plots
        if params.plots and t > 2:
            tb.plot_timeseries(t-1, N_history, penc_history, pint_history, 
                               mean_penc_history, mean_pint_history, 
                               mean_pc_history, hr_radius_history, 
                               alive_history)
        
        # If less than mim_pop or time > 1000 days, stop simulation
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
        
    tb.plot_timeseries(t, N_history, penc_history, pint_history, 
                       mean_penc_history, mean_pint_history, 
                       mean_pc_history, hr_radius_history, 
                       alive_history, params.beta_mean, params.beta_var)
    
    if t + 1 == params.t_max:
        print('Time s up, eradication not complete, {0} possums left after {1} days.'.format(
            pop.pop_size, params.t_max))

if __name__ == "__main__":
    main()