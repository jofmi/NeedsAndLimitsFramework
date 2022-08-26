import agentpy as ap
import random as rd
import numpy as np
import warnings

from scipy.optimize import minimize, linprog
from .tools import Uncertainty, Distribution 
from .tools import expand_args, make_list


class Individual(ap.Agent):
    """ A human individual within the Needs & Limits framework. """
    
    def setup(self):

        # Dimensional attributes
        for x in 'darec':
            setattr(self, f'n{x}', 0)
            setattr(self, f'{x}_keys', [])
            
        # Empty value arrays
        arrays = (
            'd_defs', 'd_grts', 'd_pgrts', 'd_sats','d_fuls',
            'd_imps', 'd_pimps',  'd_weights', 'd_pweights', 
            'a_ints', 'r_defs', 'r_stocks', 'r_flows',
            'e_defs', 'e_flows')
        for a in arrays:
            setattr(self, a, np.array([]))
            
        # Activities
        self.a_bounds = []
        self.a_ints_from_choice = None
        for x in 'dre':  
            for p in ('','p'):
                setattr(self, f'a{x}_{p}impacts', np.empty((0,0)))
            
        # Choices
        self.c_vals = []
        self.c_impacts = []
        self.c_options = []

        # Quality of life
        self.qol_sub = self.qol_psub = self.qol = self.pqol = None
        
        # Reference dictionaries
        self.key_to_ind = {}
        self.key_to_dim = self.model.key_to_dim
        
        # Performance settings
        if self.p.clip and self.p.d_defs == False:
            self.calc_pqol = self.calc_pqol_no_clip_no_defs
        elif self.p.clip == False:
            self.calc_pqol = self.calc_pqol_no_clip
        elif self.p.d_defs == False:
            self.calc_pqol = self.calc_pqol_no_defs
        
    
    # Adding dimensions (helpers) ------------------------------------------ #
    
    def update_weights(self): 
        """ Update domain weights from importance factors. """
        self.d_weights = self.d_imps / np.sum(self.d_imps)
        self.d_pweights = self.d_pimps / np.sum(self.d_pimps)
    
    def _new_keys(self, keys, dim):
        keys = make_list(keys)
        for key in keys:
            self._new_key(key, dim)
            
    def _new_key(self, key, dim):
        """ Assign new entry to a given dimension. """
        if key in self.model.key_to_dim:
            dim2 = self.model.key_to_dim[key]
            if dim2 != dim:
                raise ValueError(f"Key {key} defined for {dim} and {dim2}.")
        else:
            self.model.key_to_dim[key] = dim
        x_keys = getattr(self, f'{dim}_keys')
        x_keys.append(key)
        nx = len(x_keys)
        setattr(self, f'n{dim}', nx)
        self.key_to_ind[key] = i = nx - 1
        return i
    
    def _extend(self, keys, values=0):
        """ Add column with value v to a list of numpy matrices. """
        keys, values = expand_args(keys, values)
        for k, v in zip(keys, values):
            a = getattr(self, k)
            setattr(self, k, np.append(a, v))
    
    def _add_cols(self, key, v=0, n=1):
        """ Add column with value v to a numpy matrix. """
        a = getattr(self, key)
        setattr(self, key, np.c_[a, np.ones((a.shape[0], n)) * v])
    
    def _add_rows(self, key, v=0, n=1):
        """ Add row with given v to a numpy matrix. """
        a = getattr(self, key)
        setattr(self, key, np.r_[a, np.ones((n, a.shape[1])) * v])
    
    
    # Adding dimensions (public) ------------------------------------------- #
    
    def add_domains(self, keys, defs=0,
                    grts=1, pgrts=None, 
                    imps=1, pimps=None, 
                    sub=None, psub=None):
        """ Add life domains to this individual. 
        
        Arguments:
            keys (list of str):
                Name of each domain to be added.
            defs (float or list of float, optional):
                Default fulfillment values (default 0).*
            grts (float or list of float, optional): 
                Growth-rates of the fulfillment functions (default 1).* 
            pgrts (float or list of float, optional):
                Perceived growth-rates (default: grts).* 
            imps (float or list of float, optional):
                Importance factors (default 1).* 
            pimps (float or list of float, optional):
                Perceived importance factors (default: imps)* 
            sub (float, optional):
                Substitution factor between domains
                (default: existing substution factor).**
            psub (float, optional)
                Perceived substitution factor (default: subst).**
           
        Notes:
            * If a single value is given instead of a list,
            the same value will be used for each domain.
            ** A single substitution factor is given for all domains.
        """
        
        # Substitution factor
        if sub is not None:
            self.qol_sub = sub
            self.qol_psub = sub if psub is None else psub
        
        # Perceived factors
        pimps = pimps if pimps is not None else imps
        pgrts = pgrts if pgrts is not None else grts
        
        # Extend matrices
        keys, defs, grts, pgrts, imps, pimps, sats, fuls = \
            expand_args(keys, defs, grts, pgrts, imps, pimps, 0, 0)  
        n = len(keys)
        self._new_keys(keys, 'd')
        self._add_rows('ad_impacts', n=n)
        self._add_rows('ad_pimpacts', n=n)
        self._extend(
            ('d_defs', 'd_imps', 'd_grts', 'd_pimps', 
             'd_pgrts', 'd_sats', 'd_fuls'),
            (defs, imps, grts, pimps, pgrts, sats, fuls))
        
        # Update dependent factors
        self.update_weights()
            
    def add_resources(self, keys, defs=0):
        """ Add resource dimensions to this individual. 
        
        Arguments:
            keys (list of str):
                Name of each resource to be added.
            defs (float or list of float): 
                Default flows of this resource.*
                
        Notes:
            * If a single value is given instead of a list,
            the same value will be used for each resource.
        """
        keys, defs, stocks, flows = expand_args(keys, defs, 0, 0)
        n = len(keys)
        self._new_keys(keys, 'r')
        self._add_rows('ar_impacts', n=n)
        self._add_rows('ar_pimpacts', n=n)
        self._extend(
            ('r_defs', 'r_stocks', 'r_flows'), 
            (defs, stocks, flows))
        
    def add_activities(self, keys, impacts):
        """ Add activities to this individual. 
        
        Arguments:
            keys (list of str):
                Name of each resource to be added.
            impacts (dict): 
                Dictionary with activity impacts.
                Keys describe impact dimension,
                values are list of float that 
                describe impact of each activity.
        """
        
        # Prepare matrices
        a_i_list = []
        for key in make_list(keys):
            a_i = self._new_key(key, 'a')
            a_i_list.append(a_i)
            self._extend('a_ints', 1)
            self.a_bounds.append((0, None))
            for x in 'dre':
                self._add_cols(f'a{x}_impacts')
                self._add_cols(f'a{x}_pimpacts')
        
        # Assign impacts
        for x_key, values in impacts.items():
            x_dim = self.key_to_dim[x_key]
            if x_key in self.key_to_ind:
                x_i = self.key_to_ind[x_key]
            elif x_dim == 'e':
                x_i = self.add_efactor(x_key)
            else:
                raise ValueError(
                    f"Dimension {x_key} not defined for agent {self.id}.")
            values = make_list(values)
            if len(values) != len(a_i_list):
                raise ValueError(
                    f"Length of impact {x_key} ({len(values)}) doesn't "
                    f"match number of added activities ({len(a_i_list)})")
            ax_impacts = getattr(self, f"a{x_dim}_impacts")
            for a_i, value in zip(a_i_list, values):
                ax_impacts[x_i, a_i] = value
    
    def add_choice(self, key, options=None, impacts=None):
        """ Add a discrete choice set to this individual. 
        
        Arguments:
            key (str):
                Name of each resource to be added.
            options (list of str):
                Name of each possible choice within the choice set.
            impacts (dict): 
                Dictionary with choice impacts.
                Keys describe impact dimension,
                values are list of float that 
                describe impact of each option.
        """
        c_i = self._new_key(key, 'c')
        self.c_vals.append(0)
        self.c_impacts.append(impacts) 
        self.c_options.append(options)
    
    def add_efactor(self, key):
        """ Add environmental factor to this individual.

        Arguments:
            key (str): 
                Name of an existing environmental factor 
                that has already been added to the model.
        """
        self._add_rows('ae_impacts')
        self._add_rows('ae_pimpacts')
        self._extend(('e_defs','e_flows'))
        return self._new_key(key, 'e')

    
    # Variable access ------------------------------------------------------ #
             
    def get(self, var_key, dim_key):
        """ Retrieve an agent variable.
        
        Arguments:
            var_key (str): Name of the variable.
            dim_key (str): Name of the dimension.
        """
        i = self.key_to_ind[dim_key]
        return getattr(self, var_key)[i]
    
    def get_impact(self, a_key, dim_key):
        """ Retrieve an impact variable.
        
        Arguments:
            a_key (str): Name of the activity.
            dim_key (str): Name of the dimension.
        """
        a_i = self.key_to_ind[a_key]
        x_i = self.key_to_ind[dim_key]
        x = self.key_to_dim[dim_key]
        return getattr(self, f'a{x}_impacts')[x_i, a_i]
    
    def set_impact(self, a_key, dim_key, value):
        """ Set an impact variable.
        
        Arguments:
            a_key (str): Name of the activity.
            dim_key (str): Name of the dimension.
            value (float): Value to be set.
        """
        a_i = self.key_to_ind[a_key]
        x_i = self.key_to_ind[dim_key]
        x = self.key_to_dim[dim_key]
        getattr(self, f'a{x}_impacts')[x_i, a_i] = value
    
    
    # Recording ------------------------------------------------------------ #
    
    def record_resources(self, stocks=True, defs=True): # gains=True, 
        """ Record resource values.
        
        Arguments:
            stocks (bool, optional): Record resource stocks (default True).
            defs (bool, optional): Record default flows (default True).
        """
        
        # TODO Improve gains
        #gains (bool, optional): Record gains (flows-defs) (default True).
        
        for r_key in self.r_keys:
            r = self.key_to_ind[r_key]
            if stocks:
                self.record('r_stock_' + r_key, self.r_stocks[r])
            #if gains:
            #    self.record(
            #        'r_gain_' + r_key, self.r_stocks[r] 
            #        - self.r_old_stocks[r] - self.r_defs)
            if defs:
                self.record('r_defs_' + r_key, self.r_defs[r])
                
    def record_activities(self, ints=True, impacts=True):
        """ Record activity values.
        
        Arguments:
            ints (bool, optional): Record activity intensities (default True).
            impacts (bool, optional): Record impacts (default True).
        """
        for a_key in self.a_keys:
            a = self.key_to_ind[a_key]
            if ints:
                self.record('a_int_' + a_key, self.a_ints[a])
            if impacts:
                for d, d_key in enumerate(self.d_keys):
                    imp = self.ad_impacts[d, a]
                    self.record(f'impact_{a_key}_{d_key}', imp)
                
    def record_domains(self, fuls=True, grts=True):
        """ Record life domain values.
        
        Arguments:
            fuls (bool, optional): Record fulfillment (default True).
            grts (bool, optional): Record growth-rates (default True).
        """
        for d_key in self.d_keys:
            d = self.key_to_ind[d_key]
            if fuls:
                self.record('d_ful_' + d_key, self.d_fuls[d])
            if grts:
                self.record('d_grt_' + d_key, self.d_grts[d])
                
    def record_choices(self, vals=True):
        """ Record choice values.
        
        Arguments:
            vals (bool, optional): Record chosen options (default True).
        """ 
        for c_key in self.c_keys:
            c = self.key_to_ind[c_key]
            if vals:
                self.record('c_val_' + c_key, self.c_vals[c])
    
    # Start of simulation (round 0) ---------------------------------------- #
    
    def apply_starting_choices(self):
        """ Apply default choices """
        for c in range(self.nc):
            diffs = self._apply_choice(c, self.c_vals[c])      
            self._choice_diffs_to_defs(diffs)
    
    # Start of each round -------------------------------------------------- #
        
    def update_stocks_and_flows(self):
        """ Apply default flows to stocks (start of round). """  
        self.d_sats = self.d_defs.copy()  # Apply default satisfaction
        self.e_flows = self.e_defs.copy()  # Apply default env. impacts 
        self.r_old_stocks = self.r_stocks.copy()  # Save old stocks
        self.r_stocks += self.r_defs  # Adapt resource stocks
    
    # Adaption phase ------------------------------------------------------- #
    
    def update_perceptions(self):
        """ Perceived impacts are last-rounds impacts. """
        self.ad_pimpacts = self.ad_impacts.copy()
        self.ar_pimpacts = self.ar_impacts.copy()
        self.ae_pimpacts = self.ae_impacts.copy()     
        
    # Discrete choices ----------------------------------------------------- #
        
    def _apply_choice(self, c, o, diffs=None, reverse=False):
        """ Calculate impact difference from discrete choice 
        on d_sats, r_stocks, and e_flows. """
        
        if diffs is None:
            diffs = {x: np.zeros(getattr(self, f'n{x}')) for x in 'dre'}
        else:
            diffs = {k: v.copy() for k, v in diffs.items()}
            
        for key, impact_list in self.c_impacts[c].items():
            impact = impact_list[o]
            dim = self.key_to_dim[key]
            i = self.key_to_ind[key]
            if reverse:
                diffs[dim][i] -= impact
            else:
                diffs[dim][i] += impact
        return diffs
        
    def _choice_diffs_to_defs(self, diffs):
        self.d_defs += diffs['d']
        self.r_defs += diffs['r']
        self.e_defs += diffs['e']
        
    def _choice_diffs_to_stocks(self, diffs, reverse=False):
        sign = -1 if reverse else +1
        self.d_sats += sign * diffs['d']
        self.r_stocks += sign * diffs['r']
        self.e_flows += sign * diffs['e']
        
    def make_choice(self, key):
        """ Choose option that results in highest QOL """
        
        # Remove effect of currently active choice
        c = self.key_to_ind[key]
        o = self.c_vals[c]  # Currently active choice
        defs = self._apply_choice(c, o, reverse=True)
        
        # Calculate hypothetical QOL for each option
        qol_per_opt, a_ints_per_opt, diffs_per_opt = [], [], []
        for o, o_key in enumerate(self.c_options[c]):
            
            # Calculate differences
            diffs = self._apply_choice(c, o, defs)
            
            # Apply differences
            self._choice_diffs_to_stocks(diffs)
            
            # Calculate expteced QOL
            a_ints = self.calc_desired_a_ints() 
            a_ints_per_opt.append(a_ints)
            qol_per_opt.append(-self.calc_pqol(a_ints))
            diffs_per_opt.append(diffs)
            
            # Remove differences
            self._choice_diffs_to_stocks(diffs, reverse=True)
        
        # Take best option
        o = self.c_vals[c] = qol_per_opt.index(max(qol_per_opt))
        diffs = diffs_per_opt[o] 
        self._choice_diffs_to_stocks(diffs)  # For current round
        self._choice_diffs_to_defs(diffs)  # For future rounds
        self.a_ints_from_choice = a_ints_per_opt[o]
    
    # Activity decision -------------------------------------------------- #
            
    def _create_constraint(self, j):
        """ Return function based on r_stocks[j]
        that returns resulting inventory from a_ints. """
        ar_impacts_j, r_stocks_j = (self.ar_impacts[j], self.r_stocks[j])
        return lambda a_ints: np.sum(ar_impacts_j * a_ints) + r_stocks_j
    
    def _create_constraints(self):
        """ Create constraint for each resource dimension. """
        constraints = []
        for j in range(self.nr):
            constraints.append({
                'type': 'ineq',
                'fun': self._create_constraint(j),
            })
        return constraints
    
    def calc_desired_a_ints(self, r_stocks=None):
        """ Calculate desired set of activity intensities
        within the current context.
        
        Returns:
            array: Desired activitiy intensities (a_ints)
        """
        
        res = minimize(
            self.calc_pqol, 
            x0=self.a_ints, 
            method='SLSQP',
            bounds=self.a_bounds, 
            constraints=self._create_constraints(),
            options=self.model.min_opts,
        )
        
        return res.x


    # Quality of life (QOL) ------------------------------------------------ #
    
    def calc_qol(self, a_ints):
        """ Takes (potential) action and returns QOL. """
        d_sats = self.d_defs + self.ad_impacts.dot(a_ints) 
        d_sats = np.clip(d_sats, 0, None)
        d_fuls = 1 - np.exp(- self.d_grts * d_sats)
        qol = (np.sum(self.d_weights * (d_fuls ** self.qol_sub)) 
               ** (1 / self.qol_sub))
        return qol, d_fuls, d_sats

    def calc_pqol(self, a_ints):
        """ Takes (potential) action and returns negative perceived QOL. """
        return (- np.sum(self.d_pweights * (
                (1 - np.exp(- self.d_pgrts * 
                (np.clip(self.d_defs + self.ad_pimpacts.dot(a_ints), 0, None))
                )) ** self.qol_psub)) ** (1 / self.qol_psub))
    
    def calc_pqol_no_clip(self, a_ints):
        """ Takes (potential) action and returns negative perceived QOL. """
        return (- np.sum(self.d_pweights * (
                (1 - np.exp(- self.d_pgrts * 
                (self.d_defs + self.ad_pimpacts.dot(a_ints))
                )) ** self.qol_psub)) ** (1 / self.qol_psub))
        
    # Action --------------------------------------------------------------- #
    
    def decide_activities(self):
        if self.a_ints_from_choice is not None:
            self.a_ints = self.a_ints_from_choice
            self.a_ints_from_choice = None
        else:
            self.a_ints = self.calc_desired_a_ints()
    
    def perform_activities(self):
        """ Performs feasible set of activities that is 
        closest to desired activity pattern. """
        
        # Apply desired activities 
        # (This can result in impossible negative inventories)
        r_flows = self.ar_impacts.dot(self.a_ints)
        self.r_stocks += r_flows
        
        # Move activity intensities into feasibility space
        # (Delta a_ints is split into positive & negative changes)
        obj = 1 / (self.a_ints + 0.000001)
        res = linprog(
            c=np.hstack((-obj, obj)), 
            A_ub=np.hstack((-self.ar_impacts, -self.ar_impacts)), 
            b_ub=self.r_stocks, 
            bounds=([(-a, 0) for a in self.a_ints] 
                    + [(0, None) for a in self.a_ints]),
            method='revised simplex'
        )
        n = len(self.a_ints)
        delta_a_ints = res.x[:n] + res.x[n:]
        self.a_ints += delta_a_ints
        
        r_flows = self.ar_impacts.dot(delta_a_ints)
        self.r_stocks += r_flows
        
        # Document resource error
        self.clipped = np.sum(np.clip(self.r_stocks, None, 0))
        
        # Environmental impacts
        self.e_flows = self.ae_impacts.dot(self.a_ints)
        for ea, key in enumerate(self.e_keys):
            em = self.model.key_to_ind[key]
            self.model.e_flows[em] += self.e_flows[ea]
        
        qol, d_fuls, d_sats = self.calc_qol(self.a_ints)
        self.pqol = - self.calc_pqol(self.a_ints)  # Behavioral/perceived QOL
        self.qol, self.d_fuls, self.d_sats = qol, d_fuls, d_sats  # Actual QOL

        

    
