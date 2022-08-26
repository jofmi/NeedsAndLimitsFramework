import math
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# Lists and matrices

def make_list(element, keep_none=False):
    """ Turns element into a list of itself
    if it is not of type list or tuple. """
    if element is None and not keep_none:
        element = []  # Convert none to empty list
    if not isinstance(element, (list, tuple, np.ndarray)):
        element = [element]
    elif isinstance(element, tuple):
        element = list(element)
    return element

def expand_args(*args):
    """ Turn args into lists of length of first arg. """
    args = list(args)
    args[0] = make_list(args[0])
    n = len(args[0])
    for i, arg in enumerate(args):
        if not isinstance(arg, (list, tuple, np.ndarray)):
            args[i] = [arg] * n
        elif len(arg) != n:
            raise ValueError(
                f"Argument {arg} does not have the" 
                f"same length as keys ({len(arg)} != {n})")
    return args

def add_col(a):
    """ Add column of zeroes to a numpy matrix. """
    return np.c_[a, np.zeros((a.shape[0], 1))]

def add_row(a):
    """ Add row of zeroes to a numpy matrix. """
    return np.r_[a, np.zeros((1, a.shape[1]))]

def lim_cycle(iterable, limit):
    """ Cycle through iterable until limit is reached. """
    iterator = zip(itertools.cycle(iterable), range(limit))
    for x, _ in iterator:
        yield x


# Heterogenous variable generation

class Uncertainty:
    """ Uncertain value that can be 
    resolved with a method of random. """
    
    def __init__(self, args=(), method='uniform'):
        self.method = method
        self.args = args
        
    def get(self, generator):
        return getattr(generator, self.method)(*self.args)


class Distribution:
    """ Value that follows a distribution over agents. """
    
    def __init__(self, distribution):
        self.iterator = itertools.cycle(distribution)
        
    def draw(self):
        return next(self.iterator)
    
    
def truncnorm(mean, std, npgenerator):
    """ Truncated normal distribution. """
    if std == 0:
        return mean
    clip_a, clip_b = 0, np.inf
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    value, = stats.truncnorm.rvs(
        a, b, loc = mean, scale = std, 
        size=1, random_state=npgenerator)
    return value

         
# Economic functions

def sigmoid(x,x0=0,k=1):
    """ Returns logistic function """
    # https://en.wikipedia.org/wiki/Logistic_function
    return 1 / (1 + math.exp(-k*(x-x0)))

def gini(x):
    """ Takes list and returns Gini coefficient """
    # By Warren Weckesser https://stackoverflow.com/a/39513799
    mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
    if np.mean(x):
        rmad = mad / np.mean(x)  # Relative mean absolute difference
        return 0.5 * rmad      
    else:
        return 0.0

def redist(x, p):
    """ Redistribute values in x by percentage p. """
    x = np.array(x)
    revenue = np.sum(x * p) / len(x)
    return x - x * p + revenue
    

# Quality of Life

def qol_to_cantril(qol):
    """ Take quality of life value [0,1] and return
    integer step of the cantril ladder {0,1,...,10}. """    
    i = -1
    qt = 0
    s = 1 / 11
    while qt <= qol:
        i += 1
        qt += s
        if i == 10:
            return 10
    return i   
        
def cantril_hist_to_qol_list(cantril_hist, multiplicator):
    """ Take list with number of agents per cantril ladder step
    and return list of qol values per agent. """
    qol_values = []
    for i, n in enumerate(cantril_hist):
        qol = 1 / 11 * i + 1 / 11 * 0.5
        for _ in range(n*multiplicator):
            qol_values.append(qol)   
    return qol_values
    
    
# Output processing

def round_list(list_, decimals):
    if isinstance(list_, list):
        return [ round(elem, decimals) for elem in list_ ]
    else:
        return list_
    
def round_results(results, decimals=2):
    df = results.variables.Human.round(decimals=decimals)
    df = df.applymap(lambda x: round_list(x, decimals))
    results.variables.Human = df
    return results