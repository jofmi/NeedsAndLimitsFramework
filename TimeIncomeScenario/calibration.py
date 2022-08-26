import sys
sys.path.append("..")  # Include parent folder

import NeedsLimitsFramework.tools as tools
from scipy.optimize import minimize
import scipy.stats as stats
import numpy as np
import datetime
import warnings


def _calibration_error(observed_qol, expected, n_samples):
    """ Calculate correlation statistics. """
    observed_qol = [tools.qol_to_cantril(x) for x in observed_qol]
    observed = [0] * 11
    for qol in observed_qol:
        observed[qol] += 1
    observed = [o / n_samples for o in observed]
    chisq, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return chisq, p


def test_calibration(Model, parameters, cal_data, n_samples, cal_vars=None):
    """ Test correlation between model results and expected outcome. 
    
    Arguments:
        Model (class): Model to be calibrated
        parameters (dict): Fixed model parameters
        cal_data (list): Expected outcome to calibrate towards
        n_samples (int): Number of samples to draw
        cal_vars (list, optional): Calibration values to be passed to Model.
    """
    
    params = {
        **parameters, 
        'agents': 100 * n_samples, 
    }
    model = Model(params, cal_vars=cal_vars)
    model.run(display=False)
    chisq, p = _calibration_error(model.agents.qol, cal_data, n_samples)
    print(f"Chi-squared test statistic = {chisq}")
    print(f"P value = {p}")    


def calibrate_model(Model, parameters, x0, cal_data, n_samples=1, xtol=None, display=True):
    """ Calibrate variables of a model towards an expected outcome.
    
    Arguments:
        Model (class): Model to be calibrated
        parameters (dict): Fixed model parameters
        x0 (list): Initial guess for calibration variables
        cal_data (list): Expected outcome to calibrate towards
        n_samples (int): Number of samples to draw per calibration step
        xtol (float): Tolerance value for calibration algorithm
        display (bool): Print calibration progress
        
    Returns:
        dict: Calibration results
    """
    
    a = datetime.datetime.now().replace(microsecond=0)
    iteration = [0]
    xmin = 0.001
    
    def test_cal_vars(cal_vars):
        """ Run model with cal_vars """
        params = {
            **parameters, 
            'agents': 100 * n_samples, 
        }
        # Ensure positive values
        if np.min(cal_vars) < 0:
            cal_vars = [max(xmin, c) for c in cal_vars]
        model = Model(params, cal_vars=cal_vars)
        model.run(display=False)
        
        # Calculate goodness of fit  
        error, p = calibration_error(model.agents.qol, cal_data, n_samples) 
        
        if display:
            iteration[0] += 1
            print(f"\rRound {iteration[0]}, error = {round(error, 2)}, " 
              f"cal_vars = {[round(v, 2) for v in cal_vars]}", end=" "*10)
        
        return error
    
    # Perform optimization
    # Powell algorithm is not gradient-based
    res = minimize(
        test_cal_vars, 
        x0=np.array(x0), 
        method='Powell',  
        bounds=[(xmin, None)] * len(x0),
        options={
            'xtol': xtol
        }
    )
    
    if display:
        print(f"\n{res.message}")
        b = datetime.datetime.now().replace(microsecond=0)
        print(f"Calculation time: {b-a}")
        
    return res