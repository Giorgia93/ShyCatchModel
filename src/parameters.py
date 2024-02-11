"""
Class defining simulation parameters
"""
import scipy.stats
from datetime import datetime, date
import pandas as pd
import numpy as np
import toolbox as tb
import os


class Params:
    def __init__(self, start_date=0, end_date=1000):
        
        
        # DATAFILES        
        res_date = "20230814"
        self.posterior_results_file = "../results/" + res_date + "/posteriors_" + res_date + ".csv"
        
        res_date = date.today().strftime("%Y%m%d")
        self.simulation_Cresults_file = "../results/" + res_date + "/model_captures_" + res_date + ".csv"
        self.simulation_C2results_file = "../results/" + res_date + "/model_captures_jumpdate_" + res_date + ".csv"
        self.simulation_Nresults_file = "../results/" + res_date + "/model_N_" + res_date + ".csv"
        self.simulation_peresults_file = "../results/" + res_date + "/model_penc_" + res_date + ".csv"
        self.simulation_hrresults_file = "../results/" + res_date + "/model_hr_" + res_date + ".csv"
        self.simulation_piresults_file = "../results/" + res_date + "/model_pint_" + res_date + ".csv"
        self.scenarios_results_file = "../results/scenarios_" + res_date
        
        # Import capture data
        self.capture_data_file = "../data/dataSummary.csv"
        self.data = tb.import_data(self.capture_data_file)
        if start_date == 0:
            start_date = self.data.date[0]
        
        # Check if the directory exists
        if not os.path.exists("../results/" + res_date):
            os.makedirs("../results/" + res_date)
        
        
        # SIMULATION CONSTANTS
        self.t0 = start_date
        self.t_max = end_date
        if isinstance(end_date, datetime):
            self.t_max = (end_date - start_date).days  # Maximum number of sim days
        self.min_pop = 2          # Minimum pop. size to declare complete erad.
        self.verbose = True
        
        
        # ABC ESTIMATION PARAMETERS
        self.n_samples = 30000
        self.pars_to_fit = {"beta_mean" : [0.1, 0.9], 
                            "beta_var" : [0.01, 0.2], 
                            "N0" : [140, 280]} 
                            # "trap_grid_spacing" : [50, 200]}
                            # "g0s_coeff1" : [1, 5],
                            # "g0s_coeff2" : [-1, -0.7]}     
        self.tolerance = 0.01   # Proportion of simulations to retain (% best)
        
        
        # TRAP GRID CONSTANTS
        self.trapping_on = True     # Boolean to switch trapping on or off
        self.trap_grid_spacing = 100.       # (m) Distances between traps (square grid)
        self.trap_locations = pd.read_csv("../data/LB_trap_locations_nztm.csv")
        self.period2_start = int((datetime.strptime("09-Jun", "%d-%b") - 
                                 self.t0).total_seconds() / (60*60*24))
        self.period3_start = int((datetime.strptime("15-Jun", "%d-%b") - 
                                 self.t0).total_seconds() / (60*60*24))
        
        # POPULATION CONSTANTS
        
        # Population size
        self.N0 = 160                       # Initial population
        self.max_pop_size = 1000            # Maximum population
        
        self.k = 9                          # Carrying capacity per ha (Warburton2009)
        self.study_area = 175               # was 176 ? (ha)
        self.K = self.k * self.study_area   # Carrying capacity
        self.Kcap = int(self.K * 1.5)       # Pop size cap for array initialisation
        self.D0 = self.N0 / self.study_area # Initial density
        
        # Life-span
        self.life_span = 13    # life span in years (Cowan 2001)

        # Mortality rate
        self.daily_mort_rate = 1 / (self.life_span * 365 + 1)    # Lustig2018
        self.annual_mort_rate = 1 / (self.life_span + 1)
          
        # Reproduction and density-dependent newborn mortality rates
        self.annual_rep_rate = 0.77         # Hone 2010b, Hickling and Pekelharing 1989b
        self.annual_growth_rate = self.annual_rep_rate - self.annual_mort_rate  # per capita contribution to population
        self.dd_mor = self.annual_growth_rate / self.K       # Density dependent mortality of newborn
        
        # Reproduction period (peak day and around it)
        self.rep_peakday_date = datetime.strptime("01-Apr", "%d-%b")
        self.rep_peakday = int((self.rep_peakday_date - self.t0).total_seconds()
                               / (60*60*24))
        self.rep_sd = 20        # All animals will reproduce between rep_peakday +- 3*rep_sd
        self.birth_prob = scipy.stats.norm(self.rep_peakday, self.rep_sd).pdf(range(365)) # Birth probability per day from normal distribution 
        # self.birth_prob = scipy.stats.norm(90, self.rep_sd).pdf(range(365)) # Birth probability per day from normal distribution 
        # np.trapz(self.birth_prob)
        # self.t0 = datetime.strptime('01/01/1900 00:00:00', '%m/%d/%Y %H:%M:%S')
        # from matplotlib import pyplot as plt
        # plt.plot(self.birth_prob)
        
        # Behaviour
        self.beta_mean = 0.3                # Mean of p_int distribution
        self.beta_var = 0.1                 # Var of p_int distribution
        self.vert_trans = 0                 # Proportion of offspring inheriting parent trap-shyness
        
        
        # INDIVIDUAL CONSTANTS
        
        # Movement params
        self.diffusion_coeff = 0.01         # (m2/second)
        self.perception_dist = 10           # (m)
        self.filter_dist_bool = False       # If true, penc=0 for traps outside hr
        
        # HR size and all possible p_encTOT
        self.max_sigma = 155                # (m) From Vattiato2023
        self.max_hr_radius = 2.45 * self.max_sigma            # (m)
        self.sD_fit_coeff = [4.30, -0.4]    # log(sigma)=a+b*Density from Vattiato2023
        self.g0s_fit_coeff = [5.67, -0.99]  # g0=a*sigma^b from Vattiato2023
        self.hr_radius0 = tb.get_home_range_radius(self.max_hr_radius, self.D0, 
                                                   self.sD_fit_coeff)
        
        self.trap_grid_type = "square_grid"
        
        if self.trap_grid_type == "from_data":
            [self.hr_centres, self.all_distances_from_traps] = \
                tb.get_distances_from_trap_data(self.trap_locations, 
                                                self.perception_dist, 
                                                self.Kcap, self.hr_radius0)
            # Get distances from traps and corresponding penc for the three periods
            self.distances_from_traps1 = \
                    tb.filter_active_traps(self.trap_locations, 
                                           self.all_distances_from_traps, 1)
            self.distances_from_traps2 = \
                    tb.filter_active_traps(self.trap_locations, 
                                           self.all_distances_from_traps, 2)
            self.distances_from_traps3 = \
                    tb.filter_active_traps(self.trap_locations, 
                                           self.all_distances_from_traps, 3)
               
            self.possible_hr_radii = \
                np.arange(int(tb.get_home_range_radius(self.max_hr_radius, 
                                                       self.Kcap, self.sD_fit_coeff)), 
                          self.max_hr_radius)
                
            self.p_enc_all = tb.get_all_p_encounter(self.possible_hr_radii, 
                                                    self.distances_from_traps1, 
                                                    self.beta_mean, self.g0s_fit_coeff)
            self.p_enc_all2 = tb.get_all_p_encounter(self.possible_hr_radii, 
                                                     self.distances_from_traps2, 
                                                     self.beta_mean, self.g0s_fit_coeff)
            self.p_enc_all3 = tb.get_all_p_encounter(self.possible_hr_radii, 
                                                     self.distances_from_traps3, 
                                                     self.beta_mean, self.g0s_fit_coeff)
            
        elif self.trap_grid_type == "square_grid":
            self.all_distances_from_traps = \
                tb.get_distances_from_traps(self.trap_grid_spacing, self.hr_radius0, 
                                            self.perception_dist, self.max_hr_radius,
                                            self.Kcap)
            self.possible_hr_radii = \
                np.arange(int(tb.get_home_range_radius(self.max_hr_radius, 
                                                       self.Kcap, self.sD_fit_coeff)), 
                          self.max_hr_radius)
                
            self.p_enc_all = tb.get_all_p_encounter(self.possible_hr_radii, 
                                                    self.all_distances_from_traps, 
                                                    self.beta_mean, self.g0s_fit_coeff)
            
            
        # PLOT PARAMS
        self.plots = False      # If true, plot live plots
        
        
        # CHANGE OF LURE
        
        # Method to simulate a change of lure. Possible values are 
        # - "bump_up": all remaining p_int get a bump up defined by 
        #              change_lure_threshold (below)
        # - "redraw": all surviving individuals get a new p_int redrawn from a
        #             new beta distribution
        self.type_of_change = "lure_prop"
        
        # If the ratio of current average daily captures to initial average
        # daily captures drops to this factor, a change of lure happens 
        self.change_lure_threshold = 0.2
        
        # If using the "bump_up" or the "redraw" change of lure methods (see
        # population.change_lure(..)), this is the factor applied to surviving
        # individuals' p_int
        self.new_lure_effectiveness = 2.5   # Multiplicative factor 
        
        # If using the "lure_prop" change of lure method, this is the 
        # proportions of traps lured with lure A (default) and lure B before
        # and after change of lure threshold is met. Default [[1, 0], [1, 0]]
        # is having lure A all the time
        self.lure_prop = [[1, 0], [1, 0]]
        
        # All possible scenarios of proportions of traps lured with 
        # lure A (default) and lure B. 
        # See toolbox.choose_lure_change_scenario(..)
        self.all_lure_prop_scenarios = [[1, 0], [0.5, 0.5], [0, 1]]
        
        
        
        