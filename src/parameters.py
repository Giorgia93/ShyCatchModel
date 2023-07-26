"""
Class defining simulation parameters
"""
import scipy.stats


class Params:
    def __init__(self):
        
        # POPULATION CONSTANTS
        self.study_area = 100               # (ha)
        self.N0 = 1000                       # Initial population
        self.D0 = self.N0 / self.study_area # Initial density
        
        # Population size
        self.max_pop_size = 1000            # Maximum population
        self.k = 9                          # Carrying capacity per ha (Warburton2009)
        self.K = self.k * self.study_area   # Carrying capacity
        self.Kcap = int(self.K * 1.5)       # Pop size cap for array initialisation
        
        # Life-span
        self.life_span = 13;    # life span in years (Cowan 2001)

        # Mortality rate
        self.daily_mort_rate = 1 / (self.life_span * 365 + 1);    # Lustig2018
        self.annual_mort_rate = 1 / (self.life_span + 1);
        
        # Reproduction and density-dependent newborn mortality rates
        self.annual_rep_rate = 0.77         # Hone 2010b, Hickling and Pekelharing 1989b
        self.annual_growth_rate = self.annual_rep_rate - self.annual_mort_rate  # per capita contribution to population
        self.dd_mor = self.annual_growth_rate / self.K       # Density dependent mortality of newborn
        
        # Reproduction period (peak day and around it)
        self.rep_peakday = 90   # End of april, mid autumn
        self.rep_sd = 20        # All animals will reproduce between rep_peakday +- 3*rep_sd
        self.birth_prob = scipy.stats.norm(self.rep_peakday, self.rep_sd).pdf(range(365)) # Birth probability per day from normal distribution 

        # Behaviour
        self.beta_mean = 0.3                # Mean of p_int distribution
        self.beta_var = 0.1                 # Var of p_int distribution
        self.vert_trans = 0                 # Proportion of offspring inheriting parent trap-shyness
        
        
        # INDIVIDUAL CONSTANTS
        
        # Maximum hr size
        self.max_sigma = 155                # (m) From Vattiato2023
        self.max_hr_radius = 2.45 * self.max_sigma            # (m)
        self.sD_fit_coeff = [4.30, -0.4]    # log(sigma)=a+b*Density from Vattiato2023
        self.g0s_fit_coeff = [5.67, -0.99]  # g0=a*sigma^b from Vattiato2023
        
        # Movement params
        self.diffusion_coeff = 0.01         # (m2/second)
        self.perception_dist = 10           # (m)
        self.filter_dist_bool = False       # If true, penc=0 for traps outside hr
        
        
        
        # TRAP GRID CONSTANTS
        
        self.trapping_on = True     # Boolean to switch trapping on or off
        
        self.trap_grid_spacing = 200.       # (m) Distances between traps (square grid)
        
        # SIMULATION CONSTANTS
        self.t_max = 1000        # Maximum number of sim days
        self.min_pop = 10      # Minimum pop. size to declare complete erad.
        
        # PLOT PARAMS
        self.plots = False      # If true, plot live plots
        