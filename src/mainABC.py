"""
Created on Thu Jul 31 15:50:00 2023
@author: VattiatoGi

SHYCATCH Model with ABC estimation
Used to simulate dynamics of wild populations of possums during an eradication 
program. Model includes adaptive home-range sizes, population dynamics, 
heterogeneous levels of trap-shyness, and change of lure.

"""
from parameters import Params
from population import Population
import numpy as np
import toolbox as tb
from copy import copy
import pandas as pd
from datetime import datetime


def main():
    # --------------- CREATE PARAMETER STRUCTURE
    params = Params()
    data = params.data
    # plots(params, data)
    
    # --------------- RUN ABC FOR PARAMETER FITTING
    # run_ABC(params, data)
    
    # --------------- INITIALISE AND RUN SCENARIOS
    # Scenario parameters initialisation
    scenarios = pd.DataFrame({"change_lure_threshold": [1, 1, 0.8, 0.95, 1, 0.8, 0.95],
                              "lure_prop_scenario": ["A", "AB", "A->B", "A->B", "AB", "A->B", "A->B"],
                              "new_lure_mean_pint_mult": [1, 1, 1, 1, 2, 2, 2]})
    
    n_scenarios = np.size(scenarios, 0) # Total number of scenarios to run
    save_res_bool = n_scenarios == 1 # Only save single scenarios if only one is run
    
    # Choose start and end date for simulations. Use =data.date[0] if fitting to data
    start_date = datetime.strptime("01-Jun", "%d-%b") 
    end_date = datetime.strptime("31-Jan", "%d-%b")
    end_date = end_date.replace(year=end_date.year + 1) # Make sure end date is following year
    
    # Create parameter structure
    params = Params(start_date, end_date)
    
    # Import posterior distributions
    posterior = pd.read_csv(params.posterior_results_file)
    # Plot histograms of posterior distribution of parameters
    tb.plot_posteriors(posterior, params.pars_to_fit)
    
    # Number of simulations if using the repeated sim function
    nsims = 10000
    # # Index of posterior set to use (minimum error (best fit) for now)
    # posterior_index = pd.Series(posterior.error).idxmin()
    
    # Initialise scenario results array
    scenario_results = np.empty((nsims, 15, n_scenarios))
    
    for iscen in np.arange(n_scenarios):
        print('Scenario {0} of {1}'.format(iscen, n_scenarios))
        
        # Update parameter structure with scenario's parameters
        params.change_lure_threshold = scenarios.change_lure_threshold[iscen]
        params.lure_prop = tb.choose_lure_change_scenario(\
                            params.all_lure_prop_scenarios, 
                            scenarios.lure_prop_scenario[iscen])     
        params.new_lure_mean_pint_mult = scenarios.new_lure_mean_pint_mult[iscen]
        
        # Run nsims number of repetitions for the current scenarios using 
        # a range of posterior parameter sets (randomly drawn from the 
        # posterior distribution, with repetition) 
        
        random_indexes = np.random.choice(range(0, np.size(posterior, 0)), size=nsims)
        # Save time to eradication, eradication bool, and final population size
        [t_erad, erad_bool, N_final, change_day, final_mean_pint, final_mean_pintB, \
            nb_captures, n_history, penc_history, hr_history, pint_history, tstar80, \
            tstar80_pop, tstar80_pint, tstar95, tstar95_pop, tstar95_pint] = \
            run_scenarios_fullposterior_repeated(params, posterior, 
                                                  random_indexes, data, 
                                                  save_res_bool)
        
        # Run nsims number of repetitions for the current scenarios using 
        # single posterior parameter set with index indicated above. 
        # Save time to eradication, eradication bool, and final population size
        # [t_erad, erad_bool, N_final, change_day, final_mean_pint, final_mean_pintB, \
        #     nb_captures, n_history, penc_history, hr_history, pint_history] = \
        #     run_scenarios_repeated(params, posterior, posterior_index, nsims, 
        #                             data, save_res_bool)
            
        # Uncomment this if wanting to run scenarios using all parameter 
        # sets in the posterior distribution, instead of only the best 
        # fitting one. Note: this will only run one simulation per 
        # parameter set:
        # run_scenarios_fullposterior(posterior, params, data, save_res_bool)
        
        # Store model outputs of interest in separate structure for saving later
        scenario_results[:, :, iscen] = np.transpose([posterior.beta_mean[random_indexes], 
                                                      posterior.beta_var[random_indexes], 
                                                      posterior.N0[random_indexes], 
                                                      t_erad, erad_bool, N_final, 
                                                  change_day, final_mean_pint,
                                                  final_mean_pintB, tstar80, 
                                                  tstar80_pop, tstar80_pint, 
                                                  tstar95, tstar95_pop, tstar95_pint])
    
        # Plot current scenario results
        tb.plot_single_scenario(params.t0, params.t_max, nb_captures, 
                                    n_history, penc_history, hr_history, 
                                    pint_history, posterior.error) 
    
    # Save scenario results
    scenarios_results_file = "../results/scenarios_20240123"
    np.save(scenarios_results_file, scenario_results)
    
    # Load scenario results and plot heatmaps to compare them (paper one is scenarios_20231115)
    scenarios_results_file = "../results/scenarios_20240123"
    scenario_results = np.load(scenarios_results_file + ".npy")
    tb.produce_summary_tables(scenario_results, scenarios)
    
    
# =========================================================================
#                             ABC MODEL FITTING
# =========================================================================

def run_ABC(params, data):
    
    # Get random samples from uniform distributions
    random_samples = tb.get_random_samples(params.pars_to_fit, params.n_samples)
    
    # Initialise array of errors for each sample
    errors = np.zeros(params.n_samples)
    
    for i in range(params.n_samples):
        
        print('Sample {0} of {1}'.format(i+1, params.n_samples))
        
        params.beta_mean = random_samples[i, 0]
        params.beta_var = random_samples[i, 1]
        params.N0 = int(random_samples[i, 2])
        # params.trap_grid_spacing = random_samples[i, 2]
        # params.g0s_fit_coeff = [random_samples[i, 3], random_samples[i, 4]]
        
        pop = Population(params)
        
        # Initialisation of result arrays
        N_history = np.zeros(params.t_max)  # Density array
        hr_radius_history = np.zeros(params.t_max)  # hr_radius data array
        penc_history = np.zeros((params.Kcap, params.t_max)) # p_enc matrix
        pint_history = np.zeros((params.Kcap, params.t_max)) # p_enc matrix
        pc_history = np.zeros((params.Kcap, params.t_max))   # p_capture matrix
        alive_history = np.ones((params.Kcap, params.t_max)) # alive/dead bool matrix
        mean_penc_history = np.zeros(params.t_max)   # p_enc matrix
        mean_pint_history = np.zeros(params.t_max)   # p_int matrix
        mean_pc_history = np.zeros(params.t_max) # p_capture matrix
        change_lure_day = float("nan")
    
        for t in range(params.t_max):
            
            # If less than min_pop or time > 1000 days, stop simulation
            if pop.pop_size < params.min_pop:
                if params.verbose:
                    print('Eradication complete in {0} days. {1} possums left.'\
                          .format(t, pop.pop_size))
                break
            elif t == params.t_max - 1 and params.verbose:
                print('Eradication not complete. {0} possums left after {1} \
                      days'.format(pop.pop_size, params.t_max))
            
            # Update population
            pop.update_population(t, params)
            
            # Update time series of mean p_enc, density, p_int, p_capt for plots
            N_history[t] = pop.pop_size
            hr_radius_history[t] = pop.hr_radius
            penc_history[:, t] = pop.p_enc
            pc_history[:, t] = pop.p_capture
            pint_history[:, t] = pop.p_int
            alive_history[:, t] = pop.alive
            if pop.pop_size >= params.min_pop:
                mean_penc_history[t] = np.mean(pop.p_enc[pop.alive == 1])
                mean_pint_history[t] = np.mean(pop.p_int[pop.alive == 1])
                mean_pc_history[t] = np.mean(pop.p_capture[pop.alive == 1])
            
            # Change lure if rate of capture falls below a set value
            avg_daily_capture = np.abs(N_history[t-7] - pop.pop_size) / 7
            if params.new_lure_effectiveness > 0 and \
               np.isnan(change_lure_day) and \
               avg_daily_capture < params.change_lure_threshold:
                   # and t >= 10
                pop.change_lure(params.new_lure_effectiveness)
                pint_history[:, t] = pop.p_int
                change_lure_day = copy(t)
                # tb.plot_pint_dist(t, pint_history, alive_history)
                
        # Calculate number of daily captures
        nb_captures = abs(np.diff(N_history, append=N_history[-1]))
        
        # Create date-nb captures dataframe
        # model_resultsOLD = tb.create_results_struc(params.t0, params.t_max, 
        #                                         nb_captures)
        
        model_results = tb.create_results_struc_jumpdate(params.t0, params.t_max, 
                                                nb_captures, data)
        
        # Store error associated with current sample
        errors[i] = tb.calc_error(model_results, data)
        
    # Filter samples to retain best x%
    to_keep = errors < np.quantile(errors, params.tolerance)
    posterior = pd.DataFrame(random_samples[to_keep], columns=params.pars_to_fit)
    posterior['error'] = errors[to_keep]
    
    # Save posterior to csv file
    posterior.to_csv(params.posterior_results_file, index=False)
    
    # Plot histograms of posterior distribution of parameters
    tb.plot_posteriors(posterior, params.pars_to_fit)
    
    
    
# =============================================================================
#                                SCENARIOS
# =============================================================================
def run_scenarios_fullposterior(posterior, params, data, save_res_bool):
    """
    This runs a single simulation for each parameter combination in the posterior

    """
    # Initialise matrix of number of daily captures (to compare with data)
    nb_captures = np.zeros((len(posterior), params.t_max))
    nb_captures_jumpdate = np.zeros((len(posterior), len(data.data_captures.dropna())))
    
    N_history_all = np.zeros((len(posterior), params.t_max))
    penc_history_all = np.zeros((len(posterior), params.t_max))
    hr_history_all = np.zeros((len(posterior), params.t_max))
    pint_history_all = np.zeros((len(posterior), params.t_max))
    
    # Initialise array of eradication times for each sample
    t_erad = np.zeros(len(posterior))
    erad_bool = np.zeros(len(posterior))
    N_final = np.zeros(len(posterior))
    change_day = np.zeros(len(posterior))
    final_mean_pint = np.zeros(len(posterior))
    
    for i in range(len(posterior)):
        print('Posterior sample {0} of {1}'.format(i+1, len(posterior)))
        
        params.beta_mean = posterior.beta_mean[i]
        params.beta_var = posterior.beta_var[i]
        params.N0 = int(posterior.N0[i])
        
        pop = Population(params)
        
        # Initialisation of result arrays
        N_history = np.zeros(params.t_max)  # Density array
        N_history[0] = pop.pop_size
        penc_history = np.zeros(params.t_max)  # p_enc array
        penc_history[0] = np.mean(pop.p_enc)
        hr_history = np.zeros(params.t_max)  # home-range array
        hr_history[0] = pop.hr_radius
        pint_history = np.zeros(params.t_max)  # p_int array
        pint_history[0] = np.mean(pop.p_int)
        change_lure_day = float("nan")
        
        for t in range(params.t_max - 1):
            # if t==0:
            #     active_hr = np.array([hrc for i, hrc in enumerate(params.hr_centres) if 
            #                   pop.alive[i]==True])
            #     tb.plot_traps_and_hr_locations(params.trap_locations, 
            #                                 active_hr, pop.hr_radius)
            
            # if pop.pop_size < 80:
            #     active_hr = np.array([hrc for i, hrc in enumerate(params.hr_centres) if 
            #                   pop.alive[i]==True])
            #     # tb.plot_traps_and_hr_locations(params.trap_locations, active_hr, pop.hr_radius)
                
            # if pop.pop_size < 20:
            #     active_hr = np.array([hrc for i, hrc in enumerate(params.hr_centres) if 
            #                   pop.alive[i]==True])
            #     # tb.plot_traps_and_hr_locations(params.trap_locations, active_hr, pop.hr_radius)
                
            # If less than min_pop or time > 1000 days, stop simulation
            if pop.pop_size < params.min_pop:
                penc_history[t+1:-1] = np.ones(params.t_max-t-2) * penc_history[t]
                hr_history[t+1:-1] = np.ones(params.t_max-t-2) * hr_history[t]
                pint_history[t:-1] = np.ones(params.t_max-t-1) * pint_history[t-1]
                break
            
            # Update population
            pop.update_population(t, params)
            
            # Update time series of mean p_enc, density, p_int, p_capt for plots
            N_history[t+1] = pop.pop_size
            penc_history[t+1] = np.mean(pop.p_enc)
            hr_history[t+1] = pop.hr_radius
            if sum(pop.alive) > 0:
                pint_history[t+1] = np.mean(pop.p_int[pop.alive==1])
            else:
                pint_history[t+1] = pint_history[t]

            # # Change lure if rate of capture falls below a set value (OLD METHOD)
            # if t >= 5 and t < params.t_max-7:
            #     avg_daily_capture = np.abs(N_history[t-5] - pop.pop_size) / 7
            #     if params.new_lure_effectiveness > 0 and \
            #        np.isnan(change_lure_day) and \
            #        avg_daily_capture < params.change_lure_threshold:
            #         pop.change_lure(params.new_lure_effectiveness, 
            #                         params.type_of_change)
            #         change_lure_day = copy(t)

            # Change lure if rate if rate of captures drops by x amount (NEW METHOD)
            if (t >= 5 and t < params.t_max-7) or params.change_lure_threshold == 0:
                original_avg_daily_captures = params.N0 - N_history[1]
                current_avg_daily_captures = np.abs(N_history[t-5] - pop.pop_size) / 7
                if params.new_lure_effectiveness > 0 and \
                   np.isnan(change_lure_day) and \
                   current_avg_daily_captures / original_avg_daily_captures < \
                       1 - params.change_lure_threshold:
                    old_mean_pint = np.mean(pop.p_int[pop.alive==1]) # For printing
                    pop.change_lure(params.new_lure_effectiveness, 
                                    params.prop_lure[1])
                    change_lure_day = copy(t)
                    
                    print('Lure changed on day {0}. when weekly captures dropped to {1}'.format(t, 7*current_avg_daily_captures))
                    print('Mean p_int change = {0:.2f} --> {1:.2f}'.format(old_mean_pint, np.mean(pop.p_int[pop.alive==1])))
            if t == change_lure_day + 7:
                prev_avg_daily_captures = np.abs(N_history[t-12] - N_history[t-5]) / 7
                new_avg_daily_captures = np.abs(N_history[t-5] - N_history[t]) / 7
                print('Change of lure increased captures by {0:g}% in the week following the change'.format(\
                      100*(new_avg_daily_captures-prev_avg_daily_captures)/prev_avg_daily_captures))
        t_erad[i] = t
        erad_bool[i] = pop.pop_size < params.min_pop
        N_final[i] = pop.pop_size
        change_day[i] = change_lure_day
        final_mean_pint[i] = np.mean(pop.p_int[pop.alive==1])
        
        # Calculate number of daily captures
        captures = abs(np.diff(N_history, append=N_history[-1]))
        nb_captures[i, :] = captures
        # nb_captures_jumpdate[i, :] = tb.create_results_struc_jumpdate(params.t0, params.t_max, 
        #                                         captures, data).model_captures
        N_history_all[i, :] = N_history
        penc_history_all[i, :] = penc_history
        hr_history_all[i, :] = hr_history
        pint_history_all[i, :] = pint_history
    
    
    if save_res_bool:
        np.savetxt(params.simulation_Cresults_file, nb_captures, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_C2results_file, nb_captures_jumpdate, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_Nresults_file, N_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_peresults_file, penc_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_hrresults_file, hr_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_piresults_file, pint_history_all, fmt="%s", delimiter=',')
    
    
    return t_erad, erad_bool, N_final, change_day, final_mean_pint, nb_captures, \
        N_history_all, penc_history_all, hr_history_all, pint_history_all
        

def run_scenarios_repeated(params, posterior, posterior_index, nsims, data, save_res_bool):
    
    
    """
    This runs n simulations for a single posterior combination of parameters
    """
    
    # Extract single posterior parameter set
    params.beta_mean = posterior.beta_mean[posterior_index]
    params.beta_var = posterior.beta_var[posterior_index]
    params.N0 = int(posterior.N0[posterior_index])
        
    # Initialise matrix of number of daily captures (to compare with data)
    nb_captures = np.zeros((nsims, params.t_max))
    nb_captures_jumpdate = np.zeros((nsims, len(data.data_captures.dropna())))
    
    # Initialise matrices of population size, p_enc, home-range size, and p_int
    # for each simulation repetition and each simulated day
    N_history_all = np.zeros((nsims, params.t_max))
    penc_history_all = np.zeros((nsims, params.t_max))
    hr_history_all = np.zeros((nsims, params.t_max))
    pint_history_all = np.zeros((nsims, params.t_max))
    
    # Initialise array of eradication times for each sample
    t_erad = np.zeros(nsims)
    erad_bool = np.zeros(nsims)
    N_final = np.zeros(nsims)
    change_day = np.zeros(nsims)
    final_mean_pint = np.zeros(nsims)
    final_mean_pintB = np.zeros(nsims)
    
    for i in range(nsims):
        print('Sample {0} of {1}'.format(i+1, nsims))
        
        pop = Population(params)
        
        # Initialisation of result arrays
        N_history = np.zeros(params.t_max)  # Density array
        N_history[0] = pop.pop_size
        penc_history = np.zeros(params.t_max)  # p_enc array
        penc_history[0] = np.mean(pop.p_enc)
        hr_history = np.zeros(params.t_max)  # home-range array
        hr_history[0] = pop.hr_radius
        pint_history = np.zeros(params.t_max)  # p_int array
        pint_history[0] = np.mean(pop.p_int)
        change_lure_day = float("nan")
        
        for t in range(params.t_max - 1):
            # If less than min_pop or time > 1000 days, stop simulation
            if pop.pop_size < params.min_pop:
                penc_history[t+1:-1] = np.ones(params.t_max-t-2) * penc_history[t]
                hr_history[t+1:-1] = np.ones(params.t_max-t-2) * hr_history[t]
                pint_history[t:-1] = np.ones(params.t_max-t-1) * pint_history[t-1]
                break
            
            # Update population
            pop.update_population(t, params)
            
            # Update time series of mean p_enc, density, p_int, p_capt for plots
            N_history[t+1] = pop.pop_size
            penc_history[t+1] = np.mean(pop.p_enc)
            hr_history[t+1] = pop.hr_radius
            if sum(pop.alive) > 0:
                pint_history[t+1] = np.mean(pop.p_int[pop.alive==1])
            else:
                pint_history[t+1] = pint_history[t]
            
        
            # Change lure if rate of captures drops by x amount (NEW METHOD)
            if (t >= 5 or params.change_lure_threshold == 0) and t < params.t_max-7:
                original_avg_daily_captures = params.N0 - N_history[1]
                current_avg_daily_captures = np.abs(N_history[t-5] - pop.pop_size) / 7
                if params.new_lure_effectiveness > 0 and \
                   np.isnan(change_lure_day) and \
                   current_avg_daily_captures / original_avg_daily_captures < \
                       1 - params.change_lure_threshold:
                    pop.change_lure(params.type_of_change, 
                                    lureB_prop=params.lure_prop[1])
                    change_lure_day = copy(t)
                    if params.verbose:
                        print('Lure changed on day {0} when weekly captures dropped from {1} to {2}'.\
                              format(t, 7*original_avg_daily_captures, 7*current_avg_daily_captures))

            # Verbose print about result of change of lure 7 days later:
            if t == change_lure_day + 7:
                prev_avg_daily_captures = np.abs(N_history[t-12] - N_history[t-5]) / 7
                new_avg_daily_captures = np.abs(N_history[t-5] - N_history[t]) / 7
                if params.verbose:
                        print('Change of lure changed captures by {0:g}% in the week following the change'.format(\
                              100*(new_avg_daily_captures-prev_avg_daily_captures)/prev_avg_daily_captures))
        t_erad[i] = t
        erad_bool[i] = pop.pop_size < params.min_pop
        N_final[i] = pop.pop_size
        change_day[i] = change_lure_day
        final_mean_pint[i] = np.mean(pop.p_int[pop.alive==1])
        final_mean_pintB[i] = np.mean(pop.p_intB[pop.alive==1])
        
        # Calculate number of daily captures
        captures = abs(np.diff(N_history, append=N_history[-1]))
        nb_captures[i, :] = captures
        # nb_captures_jumpdate[i, :] = tb.create_results_struc_jumpdate(params.t0, params.t_max, 
        #                                         captures, data).model_captures
        N_history_all[i, :] = N_history
        penc_history_all[i, :] = penc_history
        hr_history_all[i, :] = hr_history
        pint_history_all[i, :] = pint_history
    
    
    if save_res_bool:
        np.savetxt(params.simulation_Cresults_file, nb_captures, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_C2results_file, nb_captures_jumpdate, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_Nresults_file, N_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_peresults_file, penc_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_hrresults_file, hr_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_piresults_file, pint_history_all, fmt="%s", delimiter=',')
    
    
    return t_erad, erad_bool, N_final, change_day, final_mean_pint, final_mean_pintB, \
        nb_captures, N_history_all, penc_history_all, hr_history_all, pint_history_all


def run_scenarios_fullposterior_repeated(params, posterior, random_indexes, data, save_res_bool):
    
    """
    This runs n simulations for n parameter sets drawn (with replacement) from
    the posteriori distribution
    """
    
    nsims = len(random_indexes)
    
    # Initialise matrix of number of daily captures (to compare with data)
    nb_captures = np.zeros((nsims, params.t_max))
    nb_captures_jumpdate = np.zeros((nsims, len(data.data_captures.dropna())))
    
    # Initialise matrices of population size, p_enc, home-range size, and p_int
    # for each simulation repetition and each simulated day
    N_history_all = np.zeros((nsims, params.t_max))
    penc_history_all = np.zeros((nsims, params.t_max))
    hr_history_all = np.zeros((nsims, params.t_max))
    pint_history_all = np.zeros((nsims, params.t_max))
    
    # Initialise array of eradication times for each sample
    t_erad = np.zeros(nsims)
    erad_bool = np.zeros(nsims)
    N_final = np.zeros(nsims)
    change_day = np.zeros(nsims)
    final_mean_pint = np.zeros(nsims)
    final_mean_pintB = np.zeros(nsims)
    tstar80 = np.zeros(nsims)
    tstar80_pop = np.zeros(nsims)
    tstar80_pint = np.zeros(nsims)
    tstar95 = np.zeros(nsims)
    tstar95_pop = np.zeros(nsims)
    tstar95_pint = np.zeros(nsims)
    
    for i in range(nsims):
        print('Sample {0} of {1}'.format(i+1, nsims))
        
        # Draw single posterior parameter set from posterior distribution:
        params.beta_mean = posterior.beta_mean[random_indexes[i]]
        params.beta_var = posterior.beta_var[random_indexes[i]]
        params.N0 = int(posterior.N0[random_indexes[i]])
        
        pop = Population(params)
        
        # Initialisation of result arrays
        N_history = np.zeros(params.t_max)  # Density array
        N_history[0] = pop.pop_size
        penc_history = np.zeros(params.t_max)  # p_enc array
        penc_history[0] = np.mean(pop.p_enc)
        hr_history = np.zeros(params.t_max)  # home-range array
        hr_history[0] = pop.hr_radius
        pint_history = np.zeros(params.t_max)  # p_int array
        pint_history[0] = np.mean(pop.p_int)
        change_lure_day = float("nan")
        
        for t in range(params.t_max - 1):
            # If less than min_pop or time > 1000 days, stop simulation
            if pop.pop_size < params.min_pop:
                penc_history[t+1:-1] = np.ones(params.t_max-t-2) * penc_history[t]
                hr_history[t+1:-1] = np.ones(params.t_max-t-2) * hr_history[t]
                pint_history[t:-1] = np.ones(params.t_max-t-1) * pint_history[t-1]
                break
            
            # Update population
            pop.update_population(t, params)
            
            # Update time series of mean p_enc, density, p_int, p_capt for plots
            N_history[t+1] = pop.pop_size
            penc_history[t+1] = np.mean(pop.p_enc)
            hr_history[t+1] = pop.hr_radius
            if sum(pop.alive) > 0:
                pint_history[t+1] = np.mean(pop.p_int[pop.alive==1])
            else:
                pint_history[t+1] = pint_history[t]
            
            # Change lure if rate of captures drops by x amount (NEW METHOD)
            if (t >= 5 or params.change_lure_threshold == 0) and t < params.t_max-7:
                original_avg_daily_captures = params.N0 - N_history[1]
                current_avg_daily_captures = np.abs(N_history[t-5] - pop.pop_size) / 7
                if params.new_lure_effectiveness > 0 and \
                   np.isnan(change_lure_day) and \
                   current_avg_daily_captures / original_avg_daily_captures < \
                       1 - params.change_lure_threshold:
                    pop.change_lure(params.type_of_change, 
                                    lureB_prop=params.lure_prop[1])
                    change_lure_day = copy(t)
                    if params.verbose:
                        print('Lure changed on day {0} when weekly captures dropped from {1} to {2}'.\
                              format(t, 7*original_avg_daily_captures, 7*current_avg_daily_captures))
                
                # Save current population and time if daily captures plateau
                if current_avg_daily_captures / original_avg_daily_captures <= 1 - 0.8 \
                    and tstar80[i] == 0:
                    tstar80[i] = copy(t)
                    tstar80_pop[i] = copy(pop.pop_size)
                    tstar80_pint[i] = np.mean(pop.p_int[pop.alive==1])
                if current_avg_daily_captures / original_avg_daily_captures <= 1 - 0.95 \
                    and tstar95[i] == 0:
                    tstar95[i] = copy(t)
                    tstar95_pop[i] = copy(pop.pop_size)
                    tstar95_pint[i] = np.mean(pop.p_int[pop.alive==1])

            # Verbose print about result of change of lure 7 days later:
            if t == change_lure_day + 7:
                prev_avg_daily_captures = np.abs(N_history[t-12] - N_history[t-5]) / 7
                new_avg_daily_captures = np.abs(N_history[t-5] - N_history[t]) / 7
                if params.verbose:
                        print('Change of lure changed captures by {0:g}% in the week following the change'.format(\
                              100*(new_avg_daily_captures-prev_avg_daily_captures)/prev_avg_daily_captures))
        t_erad[i] = t
        erad_bool[i] = pop.pop_size < params.min_pop
        N_final[i] = pop.pop_size
        change_day[i] = change_lure_day
        final_mean_pint[i] = np.mean(pop.p_int[pop.alive==1])
        final_mean_pintB[i] = np.mean(pop.p_intB[pop.alive==1])
        
        # Calculate number of daily captures
        captures = abs(np.diff(N_history, append=N_history[-1]))
        nb_captures[i, :] = captures
        # nb_captures_jumpdate[i, :] = tb.create_results_struc_jumpdate(params.t0, params.t_max, 
        #                                         captures, data).model_captures
        N_history_all[i, :] = N_history
        penc_history_all[i, :] = penc_history
        hr_history_all[i, :] = hr_history
        pint_history_all[i, :] = pint_history
    
    
    if save_res_bool:
        np.savetxt(params.simulation_Cresults_file, nb_captures, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_C2results_file, nb_captures_jumpdate, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_Nresults_file, N_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_peresults_file, penc_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_hrresults_file, hr_history_all, fmt="%s", delimiter=',')
        np.savetxt(params.simulation_piresults_file, pint_history_all, fmt="%s", delimiter=',')
    
    
    return t_erad, erad_bool, N_final, change_day, final_mean_pint, final_mean_pintB, \
        nb_captures, N_history_all, penc_history_all, hr_history_all, pint_history_all, \
            tstar80, tstar80_pop, tstar80_pint, tstar95, tstar95_pop, tstar95_pint
# =============================================================================
#                                PLOTS
# =============================================================================
def plots(params, data):
    
    # Import posterior distributions
    posterior = pd.read_csv(params.posterior_results_file)
    # Plot beta distribution for best fit
    best_fit_index = posterior.error.idxmin()
    # tb.plot_beta_distribution(posterior.beta_mean[best_fit_index],
    #                           posterior.beta_var[best_fit_index])
    # tb.plot_beta_distribution(np.median(posterior.beta_mean),
    #                           np.median(posterior.beta_var))
    # tb.plot_posteriors(posterior, params.pars_to_fit)
    
    # Import model results using posterior
    nb_captures = np.genfromtxt(params.simulation_Cresults_file, delimiter=',')
    nb_captures_jumpdate = np.genfromtxt(params.simulation_C2results_file, delimiter=',')
    n_history = np.genfromtxt(params.simulation_Nresults_file, delimiter=',')
    penc_history = np.genfromtxt(params.simulation_peresults_file, delimiter=',')
    hr_history = np.genfromtxt(params.simulation_hrresults_file, delimiter=',')
    pint_history = np.genfromtxt(params.simulation_piresults_file, delimiter=',')
    
    # Plot comparison of model simulations and data
    tb.plot_captures_comparison(params.t0, params.t_max, nb_captures, 
                                nb_captures_jumpdate,
                                n_history, penc_history, hr_history, 
                                pint_history, data, posterior.error) 
    

if __name__ == "__main__":
    main()