import sys
import numpy as np
import pandas as pd
import agentpy as ap
import networkx as nx

# Import Needs & Limits Framework
sys.path.append("..")  # Include parent folder
from NeedsLimitsFramework.model import Model as Model_
from NeedsLimitsFramework.agents import Individual as Individual_
from NeedsLimitsFramework.tools import *


class Individual(Individual_):
    
    def prepare_dimensions(self):
        """ Setup of life domains, resources, activities, 
        choices, and additional variables. """
        
        # Define dimensions
        r_keys = ['money', 'time']
        r_defs = [0, 1]
        d_keys = ['needs_ma', 'needs_im', 'norms']
        d_imps = [1, 1, 1]
        d_pimps = [1, 1, self.p.imp_norms]
        
        # Draw random growth rates based on calibration variables
        d_grts = [truncnorm(mean, std, self.model.nprandom) 
                  for (mean, std) in zip(*[iter(self.model.cal_vars)]*2)]
        
        a_keys = ['cons_brown', 'cons_green', 'recreation']
        p = - 1 - self.p.ctax  # Price affected by carbon tax
        a_impacts = {
            'money':      [ p, -1,  0],
            'time':       [ 0, -1, -1],
            'needs_ma':   [ 1,  1,  0],
            'needs_im':   [ 0,  0,  1],
            'norms':      [ 0,  0,  0],  # Place-holder
            'emissions':  [ 1,  0,  0],
        }
        
        # Add dimensions
        self.add_domains(d_keys, imps=d_imps, pimps=d_pimps, grts=d_grts,
                        sub=self.p.domain_substitution)
        self.add_resources(r_keys, defs=r_defs)
        self.add_activities(a_keys, impacts=a_impacts)
        
        # Additional variables
        self.neighbors = self.model.network.neighbors(self).to_list()
        self.inc_share = self.model.inc_shares.draw()
        self.norms_pweights = [1, 1, self.p.imp_norms_recreation]
        self.norms_weights = [1, 1, 1]
        
        # Discrete choices
        c_key = 'work_time'
        c_options = ['full-time', 'part_time']
        work_times = [- hours/(24*7) for hours in (40, 30)]  
        work_incomes = [self.inc_share * self.model.income * (-wt) 
                        for wt in work_times]
        c_impacts = {
            'money': work_incomes,
            'time':  work_times
        }
        self.add_choice(c_key, options=c_options, impacts=c_impacts)
        
      
    def update_impacts_norms(self):
        """ Update of impacts of activities on social norms. """
        
        # Impacts on social norms
        d = self.key_to_ind['norms']
            
        for a, w, pw in zip(range(self.na), 
                            self.norms_weights, 
                            self.norms_pweights):
            
            # Perceived activity intensities from peers
            a_ints = [i[a] for i in self.neighbors.a_ints]
            a_int_mean = sum(a_ints) / len(a_ints)
            
            # Actual and perceived activity impacts
            self.ad_impacts[d, a] = w * a_int_mean
            self.ad_pimpacts[d, a] = pw * a_int_mean
            
    
    def update_ces(self):
        """ Update of CES Utility for current activity intensities. """
        r_used = - self.ar_impacts.dot(self.a_ints) # np.clip(, 0, None)
        ces = (np.sum(0.5 * (r_used ** self.qol_sub)) 
               ** (1 / self.qol_sub))
        self.ces = ces
        
        
class Model(Model_):
    """ 
    Time and Income Scenario. 
    An agent-based model within the N&L framework.
    
    Arguments:
        cal_vars (list, optional): 
            List of values for calibration variables.
            If none, values are taken from path.
        path (str, optional):
            Path of cal_vars, if none are passed
            (default 'data/cal_vars.txt').
    """
    
    def setup(self, cal_vars=None, path='data/cal_vars.txt'):
        
        super().setup()
        
        # Scenario labels
        self.labels = {
            'cons_brown': 'Brown consumption',
            'cons_green': 'Green consumption',
            'recreation': 'Recration & rest',
            'perc_part_time': 'Part-time choice'
        }
        
        # Load calibration variables
        self.cal_vars = np.loadtxt(path) if cal_vars is None else cal_vars
        
        # Prepare income distribution
        inc_shares = pd.read_csv(f'data/income_shares_2016.csv')['value']
        redist_strength = self.p.redist + self.p.ctax * self.p.ctax_redist
        inc_shares = redist(inc_shares, redist_strength)
        self.inc_shares = Distribution(inc_shares)
        self.income = self.p.income * (1 + self.p.growth) 
        
        # Initiate agents & environment
        self.agents = ap.AgentList(self, self.p.agents, Individual) 
        self.add_efactors(keys='emissions')
        
        # Generate agent peer network
        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.network_neighbors,
            self.p.network_randomness,
            seed = self.random.getrandbits(124))
        self.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)
        
        # Prepare agents
        self.agents.prepare_dimensions()
        self.agents.update_impacts_norms()  # Scenario method
        self.agents.apply_starting_choices()

        
    def step(self):
        
        # Start of round
        self.update_e_flows()
        self.agents.update_stocks_and_flows()
        
        # Adaption phase
        self.agents.update_perceptions()
        self.agents.update_impacts_norms()  # Scenario method
        
        # Decision phase
        self.agents.make_choice('work_time')
        
        self.agents.decide_activities()
        
        # Action phase
        self.agents.perform_activities()
        
        # End of round
        self.update_e_stocks()
        
        # Documentation
        self.agents.income = self.agents.get('r_defs', 'money')
        self.agents.update_ces()  # Scenario method
        self.agents.record(['qol', 'ces', 'income'])
        self.agents.record_activities()
        self.agents.record_resources()
        self.agents.record_domains()
        self.agents.record_choices()
        self.record('Emissions', self.e_flows[0])
            
        
    def end(self):
        
        # Aggregate reporters
        n = self.p.agents
        
        # Quality of Life
        Q = np.array(self.agents.qol)
        self.report('M(Q)', sum(Q) / n)
        self.report('G(Q)', gini(Q))
        
        Qperc = np.array(self.agents.pqol)
        self.report('M(Qperc)', sum(Qperc) / n)
        
        # CES Utility
        CES = np.array(self.agents.ces)
        self.report('M(CES)', sum(CES) / n)
        self.report('G(CES)', gini(CES))
        
        # Domain fulfillment
        for d, key in enumerate(self.agents[0].d_keys):
            MN = np.array([i.d_fuls[d] for i in self.agents])
            self.report(key, np.sum(MN) / n)
            
        # Activities
        for a, key in enumerate(self.agents[0].a_keys):
            MN = np.array([i.a_ints[a] for i in self.agents])
            self.report(key, np.sum(MN) / n)
            
        # Part time choice
        self.report(
            'perc_part_time', sum([c[0] for c in self.agents.c_vals])/n)
        
        # Income
        I = np.array(self.agents.income) 
        self.report('M(I)', np.sum(I) / n)
        self.report('G(I)', gini(I))
        
        # Emissions
        E = np.array([i.e_flows[0] for i in self.agents])
        self.report('M(E)', np.sum(E) / n)
        self.report('G(E)', gini(E))