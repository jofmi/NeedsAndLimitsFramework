import agentpy as ap
import numpy as np
import warnings
import copy

# Tools -------------------------------------------------------------------- #

from .tools import expand_args, make_list

# Matrix extensions
# Assume each row has same length

def add_row(m): 
    m.append([0] * len(m[0]))

# Main model --------------------------------------------------------------- #

class Model(ap.Model):
    """ Agent-based model with human individuals. """
    
    def setup(self):
        
        # Default settings
        self.min_opts = {
        #    'ftol': 1e-1,
        #    'maxiter': 'ftol': 1e-,
        }  
        
        # Default parameters
        pdef = {
            'clip': True,
            'd_defs': True
        }
        for k, v in pdef.items():
            if k not in self.p:
                self.p[k] = v
        
        # Initiate variables        
        self.ni = 0  # Number of agents
        self.ne = 0  # Number of env. dim. 
        self.e_keys = []  # Names of env. dim.          
        self.e_defs = np.array([])
        self.e_flows = np.array([])
        self.e_stocks = np.array([])
        
        # Reference dictionaries
        self.key_to_dim = {}
        self.key_to_ind = {}

    
    # Environmental dynamics ----------------------------------------------- #
    
    def update_e_flows(self):
        """ Update environmental flows (start of round). """
        self.e_flows = np.copy(self.e_defs)
    
    def update_e_stocks(self):
        """ Update environmental stocks (end of round). """
        self.e_stocks += self.e_flows
    
    # Adding dimensions ---------------------------------------------------- #
    
    def add_efactors(self, keys, stocks=0, defs=0):
        """ Add environmental factors to this model. 

        Arguments:
            keys (list of str):
                Name of each resource to be added.
            stocks (float or list of float): 
                Stocks at the beginning of the simulation.*
            defs (float or list of float): 
                Default flows of this environmental factor.*

        Notes:
            * If a single value is given instead of a list,
            the same value will be used for each factor.
        """
        
        keys, stocks, defs = expand_args(keys, stocks, defs)
        for key, stock, defs in zip(keys, stocks, defs):
            self.ne += 1

            self.e_keys.append(key)
            self.e_stocks = np.append(self.e_stocks, stock)
            self.e_flows = np.append(self.e_flows, 0)
            self.e_defs = np.append(self.e_defs, defs)
                        
            self.key_to_dim[key] = 'e'
            self.key_to_ind[key] = len(self.e_keys) - 1

   