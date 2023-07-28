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
        
        # Vertical transmission of trap-shyness
        self.vertical_trans = params.vert_trans
        
        # Home-range size and probability of encounter
        self.max_hr_radius = params.max_hr_radius
        self.hr_radius = self.get_home_range_radius(self.pop_density, 
                                                    params.sD_fit_coeff)
        self.distances_from_traps = tb.get_distances_from_traps(params.trap_grid_spacing, 
                                                self.hr_radius, 
                                                params.perception_dist*2, 
                                                params.max_hr_radius, params.Kcap)
        self.p_enc = self.get_p_encounter(self.distances_from_traps, 
                                          params.beta_mean,
                                          params.g0s_fit_coeff, 
                                          params.filter_dist_bool)
        
        # Probability of capture
        self.p_capture = self.p_enc * self.p_int
        
        # Individual status
        self.alive = np.concatenate([np.ones(params.N0), 
                                     np.zeros(params.Kcap - params.N0)])  
        
    
    def change_lure(self, effectiveness_change):
        """
        Function that updates the animals' p_int values according to a given
        effectiveness increase factor resulting from a change of lure

        Parameters
        ----------
        effectiveness_change : float
            Effectiveness change factor

        """
        new_pint = self.p_int + effectiveness_change
        new_pint[new_pint > 1] = 1
        self.p_int = new_pint
        
    
    
    def get_home_range_radius(self, pop_density, sD_fit_coeff):
        """
        Function that gets animals' home range radius according to given 
        population densities, using log-log lm coefficients defined in params.py
        
        Parameters
        ----------
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
        hr_radius = min(self.max_hr_radius, np.sqrt(new_hr_area / np.pi))
        
        return hr_radius
    
    
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
        
        # At start of reproduction season, reset nb of juvies and calculate
        # this year's birth rate
        if np.mod(t, 365) == params.rep_peakday - 3 * params.rep_sd:    
            self.juvies = 0     # Reset number of juvies
            self.daily_birth_rate = self.get_birth_rate(params.annual_rep_rate,
                                                        params.dd_mor, 
                                                        params.birth_prob)
        
        # Natural deaths
        nat_deaths = np.random.binomial(sum(self.alive), params.daily_mort_rate);
        if nat_deaths > 0:
            available_indexes = np.where(self.alive == 1)[0]
            dead_indexes = available_indexes[0:nat_deaths]
            self.alive[dead_indexes] = 0
            self.pop_size = max(0, self.pop_size - nat_deaths)
            
        # Trap captures
        captured_indexes = (np.random.rand(params.Kcap) <= self.p_capture) * params.trapping_on
        if sum(captured_indexes) > 0:
            self.alive[captured_indexes == 1] = 0
            self.pop_size = max(0, sum(self.alive))
        
        # Calculate today's number of newborns and update population
        nb_newborn_today = np.random.binomial(sum(self.alive), self.daily_birth_rate[np.mod(t, 365)])
        if nb_newborn_today > 0:
            self.reproduce(t, nb_newborn_today, params.beta_mean, params.beta_var)
            
        # Update current density with new pop size
        self.pop_density = self.pop_size / params.study_area
        
        # Update hr_radius with new density
        self.hr_radius = self.get_home_range_radius(self.pop_density, 
                                                    params.sD_fit_coeff)  
        
        # Update p_enc with new hr radius
        self.p_enc = self.get_p_encounter(self.distances_from_traps, 
                                          params.beta_mean,
                                          params.g0s_fit_coeff, 
                                          params.filter_dist_bool)
        # Update p_capture with new p_enc
        self.p_capture = self.p_enc * self.p_int
                                 

    def print_pop_size(self):
        print('Current population size: {0}'.format(self.pop_size))
        
    def print_hr_radius(self):
        print('Current home range radius: {0:.2f}'.format(self.hr_radius))
        
