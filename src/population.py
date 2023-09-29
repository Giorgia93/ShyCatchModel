# -*- coding: utf-8 -*-
"""
Class defining population of animals
"""

import numpy as np
import toolbox as tb


class Population:
    def __init__(self, params):
        # Population size parameters
        self.pop_size = params.N0
        self.juvies = 0
        self.pop_density = params.D0
        self.max_pop_size = params.max_pop_size
        self.daily_birth_rate = self.get_birth_rate(params.annual_rep_rate,
                                                    params.dd_mor, 
                                                    params.birth_prob)
        
        # Trap-shyness distribution parameters
        self.beta_mean = params.beta_mean
        self.beta_var = params.beta_var
        self.p_int = tb.get_p_interaction(params.Kcap, 
                                          self.beta_mean, self.beta_var)
        self.p_intB = tb.get_p_interaction(params.Kcap, 
                                          self.beta_mean * params.new_lure_mean_pint_mult, 
                                          self.beta_var)
        # Proportion of the population that display, respectively, a decreased
        # attraction towards the second lure, no change in attraction, and an
        # increase in attraction. These need to sum to one.
        self.lure_preference_prop = [0.25, 0.6, 0.15] 
        # Multipliers applied to p_int in each of the three cases described above
        self.attraction_change_mult = [0.8, 1, 3]
        # Multipliers randomly assigned to each individual with weights defined above
        self.new_lure_reaction = np.random.choice(self.attraction_change_mult, 
                                                  params.Kcap, 
                                                  p=self.lure_preference_prop) 
        # Probability of finding lure A vs lure B
        self.lure_prop = params.lure_prop[0]
        
        # Vertical transmission of trap-shyness
        self.vertical_trans = params.vert_trans
        
        # Home-range size and probability of encounter
        # self.max_hr_radius = params.max_hr_radius
        self.hr_radius = params.hr_radius0
        # [self.hr_centres, self.all_distances_from_traps] = \
        #     tb.get_distances_from_trap_data(params.trap_locations, 
        #                                     params.perception_dist, 
        #                                     params.Kcap, self.hr_radius)
        # self.distances_from_traps = \
        #     tb.filter_active_traps(params.trap_locations, 
        #                            self.all_distances_from_traps, 1)
            
        # Get all penc with current trap layout
        self.p_enc_all = params.p_enc_all
        # Only get penc for current hr size
        self.p_enc = \
            np.squeeze(self.p_enc_all[params.possible_hr_radii==int(self.hr_radius), :])
        
        # Probability of capture
        self.p_capture = self.p_enc * self.p_int
        
        # Individual status
        self.alive = np.concatenate([np.ones(params.N0), 
                                     np.zeros(params.Kcap - params.N0)])  
        
    
    def change_lure(self, type_of_change=0, effectiveness_change=0, lureB_prop=0):
        """
        Function that updates the animals' p_int values according to a given
        effectiveness increase factor resulting from a change of lure

        Parameters
        ----------
        effectiveness_change : float
            Effectiveness change factor

        """
        if type_of_change == "bump_up":
            new_pint = self.p_int + effectiveness_change
            new_pint[new_pint > 1] = 1
            self.p_int = new_pint
        elif type_of_change == "redraw":
            # Get mean and variance of surviving animals' p_int
            current_pint_mean = np.mean(self.p_int[self.alive==1])
            current_pint_var = np.var(self.p_int[self.alive==1])
            # Re-draw surviving animals p_int from beta distribution with 
            # bumped-up mean and current variance
            new_pint = tb.get_p_interaction(int(self.pop_size), 
                                            current_pint_mean * effectiveness_change, 
                                            current_pint_var)
            self.p_int[self.alive==1] = new_pint
        elif type_of_change == "lure_prop":
            self.lure_prop = lureB_prop
            
        elif type_of_change == 0 or \
            (type_of_change != "lure_prop" and effectiveness_change == 0) or \
            (type_of_change == "lure_prop" and lureB_prop == 0):
            print("Error: type of lure change not recognised")
    
    
    def get_p_encounter(self, distances, beta_mean, g0s_fit_coeff, 
                        filter_dist_bool):
        """
        Function that calculates the probability of encountering a trap on one 
        night, for a set home-range radius and for a set square grid of traps, 
        for each simulated individual
        
        Parameters
        ----------
        distances : float array
            Array of distances between home-range centres and traps
        beta_mean : float
            Mean of original p_int beta distribution
        g0s_fit_coeff : 1x2 float array
            Fitted coefficients from the power law g0 = a*sigma^b
        mean_pint : float
            Mean population probability of interaction
        filter_dist_bool : bool
            Boolean defining the probability of encounter of traps outside the HR

        Returns
        -------
        p_enc_tot : float array
            Nightly probability of encounter for each individual

        """
        
        # Sigma parameter of the half normal, calculated from given home-range size 
        # using the variance formula of the half-normal distribution
        sig = self.hr_radius / 2.45;
        
        # Probability of encounter at distance = 'dist', p_enc <= 1 (animal will 
        # cover entire home-range for small enough home-ranges and high enough 
        # diffusion coefficients
        p_enc = ((g0s_fit_coeff[0] * (sig ** g0s_fit_coeff[1])) * 
                  np.exp(-(distances ** 2) / (2 * (sig ** 2)))) / beta_mean 
        p_enc[p_enc > 1] = 1
        
        # Set p_enc for any trap beyond home-range to 0 if required
        if filter_dist_bool:
            p_enc = p_enc[distances > self.hr_radius] = 0

        # Calculate total p_enc taking in consideration all traps within home 
        # range
        p_enc_tot = 1 - np.prod(1 - p_enc, axis=0)
        
        return p_enc_tot
            
            
    def reproduce(self, t, nb_newborn_today, original_beta_mean, 
                  original_beta_var):
        """
        Function that calculates number of newborn at time t and updates
        population parameters accordingly.

        Parameters
        ----------
        t : int
            Current day (time)
        original_beta_mean : float
            Mean of p_int distribution of the original population
        original_beta_var : float
            Mean of p_int distribution of the original population

        """
            
        # Generate indexes of newborns
        available_indexes = np.where(self.alive == False)[0]
        newborn_indexes = available_indexes[0:nb_newborn_today]
        
        # Find which newborns get personality from parent distributions
        transmitted_personality = np.random.rand(nb_newborn_today) <= self.vertical_trans;
        
        # Update p_int distribution of current population
        self.update_pint_distribution()
        
        # Assign p_int to newborn, either drawn from current population
        # distribution or from original population distribution
        self.p_int[newborn_indexes[0:sum(transmitted_personality)]] = tb.get_p_interaction(sum(transmitted_personality), 
                                              self.beta_mean, self.beta_var)
        self.p_int[newborn_indexes[sum(transmitted_personality):]] = tb.get_p_interaction(sum(~transmitted_personality), 
                                          original_beta_mean, 
                                          original_beta_var)
        
        self.alive[newborn_indexes] = 1    # Change status of newborns to alive
        self.pop_size += nb_newborn_today  # Update population size
        self.juvies += nb_newborn_today    # Update num of juvies
        
            
            
    def get_birth_rate(self, annual_rep_rate, dd_mor, birth_prob):
        """
        Function that returns the new daily birth rate given the annual 
        reproduction rate, the density-depended mortality, and the seasonal
        birth probability

        Parameters
        ----------
        annual_rep_rate : float
            Annual reproduction rate
        dd_mor : float
            Density-dependent mortality rate
        birth_prob : float
            Daily birth probability from seasonal curve

        Returns
        -------
        daily_birth_rate : float
            Daily birth rate

        """
        # Calculate density dependent birth rate for the year
        cst_birth_rate = max(0, annual_rep_rate - dd_mor * self.pop_size)
        
        # Scale birth rate with reproduction season curve
        daily_birth_rate = cst_birth_rate * birth_prob
        
        return daily_birth_rate
        
        
    def update_pint_distribution(self):
        """
        Function that updates mean and variance of the p_int ditribution for 
        alive individuals.

        """
        self.beta_mean = np.mean(self.p_int[self.alive == True])
        self.beta_var = np.var(self.p_int[self.alive == True])
        
    
    def update_population(self, t, params):
        """
        Function that updates population parameters such as population size,
        home-range size, probability of encountering a trap, probability
        of capture, alive/dead status. This is done after calculating
        daily births, natural deaths, animals trapped.

        Parameters
        ----------
        t : float
            Current time step (day)
        params : struct
            Structure of parameters as defined in parameters.py

        Returns
        -------
        None.

        """
        
        # At start of reproduction season, reset nb of juvies and calculate
        # this year's birth rate
        if np.mod(t, 365) == params.rep_peakday - 3 * params.rep_sd:    
            self.juvies = 0     # Reset number of juvies
            self.daily_birth_rate = self.get_birth_rate(params.annual_rep_rate,
                                                        params.dd_mor, 
                                                        params.birth_prob)
            
        # At the start of period 2, activate second group of traps
        if params.trap_grid_type == "from_data" and \
            np.mod(t, 365) == params.period2_start: 
            # Update all penc with new traps
            self.p_enc_all = params.p_enc_all2
            self.p_enc = \
                np.squeeze(self.p_enc_all[params.possible_hr_radii==int(self.hr_radius), :])
            # Update p_capture with new p_enc
            self.p_capture = self.p_enc * self.p_int
        
        # At the start of period 3, activate third group of traps
        elif params.trap_grid_type == "from_data" and \
            np.mod(t, 365) == params.period3_start: 
            # Update all penc with new traps
            self.p_enc_all = params.p_enc_all3
            self.p_enc = \
                np.squeeze(self.p_enc_all[params.possible_hr_radii==int(self.hr_radius), :])
            # Update p_capture with new p_enc
            self.p_capture = self.p_enc * self.p_int
        
        # Natural deaths
        nat_deaths = np.random.binomial(sum(self.alive), params.daily_mort_rate);
        if nat_deaths > 0:
            available_indexes = np.where(self.alive == 1)[0]
            dead_indexes = available_indexes[0:nat_deaths]
            self.alive[dead_indexes] = 0
            self.pop_size = max(0, self.pop_size - nat_deaths)
            
            
        # Trap captures
        
        # Boolean for individuals that have found lure B (new), given the 
        # current proportion of traps lured with lure B
        lureBfound = np.random.choice([0, 1], size=params.Kcap, 
                                      p=self.lure_prop) # Individual probability of finding lure B
        
        # OLD METHOD
        # # Initialisation of pint multipliers
        # pint_multipliers = np.ones(params.Kcap) 
        # # Change multipliers to individual reactions for individuals that found the new lure
        # pint_multipliers[lureBfound == 1] = self.new_lure_reaction[lureBfound == 1]
        
        # # Update p_capture with new multipliers
        # old_mean_pcap = np.mean(self.p_capture[self.alive==1])
        # self.p_capture *= pint_multipliers
        # cur_mean_pcap = np.mean(self.p_capture[self.alive==1])
        
        # if params.verbose and self.lure_prop != [1, 0]:
        #     print('Mean p_capture change = {0:.5f} --> {1:.5f}'.format(old_mean_pcap, cur_mean_pcap))

        # NEW METHOD
        # Update p_capture with p_intB instead of original p_int
        self.p_capture[lureBfound == 1] = \
            self.p_capture[lureBfound == 1] / self.p_int[lureBfound == 1] * \
            self.p_intB[lureBfound == 1]      
        
        # Randomly draw newly captured individuals, if trapping is on
        captured_indexes = (np.random.rand(params.Kcap) <= self.p_capture) * params.trapping_on
        if sum(captured_indexes) > 0:
            self.alive[captured_indexes == 1] = 0    # Update alive individuals
            self.pop_size = max(0, sum(self.alive))  # Update pop size
        
        # Calculate today's number of newborns and update population
        nb_newborn_today = np.random.binomial(sum(self.alive), self.daily_birth_rate[np.mod(t, 365)])
        if nb_newborn_today > 0:
            self.reproduce(t, nb_newborn_today, params.beta_mean, params.beta_var)
            
        # Update current density with new pop size
        self.pop_density = self.pop_size / params.study_area
        
        if self.pop_size > 0:
            # Update hr_radius with new density
            self.hr_radius = tb.get_home_range_radius(params.max_hr_radius, 
                                                      self.pop_density, 
                                                      params.sD_fit_coeff)  
            # Update p_enc with new hr radius
            self.p_enc = \
                np.squeeze(self.p_enc_all[params.possible_hr_radii==int(self.hr_radius), :])
            
            # Update p_capture with new p_enc
            self.p_capture = self.p_enc * self.p_int
                                 

    def print_pop_size(self):
        print('Current population size: {0}'.format(self.pop_size))
        
    def print_hr_radius(self):
        print('Current home range radius: {0:.2f}'.format(self.hr_radius))
        
