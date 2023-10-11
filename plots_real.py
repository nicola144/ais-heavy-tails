import matplotlib.pyplot as plt # plotting

import pickle
import numpy as np
import requests
import sys
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import rpy2py
import rpy2.robjects as ro
import json
import jax
from collections import OrderedDict
import gc
import os
import matplotlib.pyplot as plt
# Enable LaTeX for nicer plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')




# load
def load_all_npy_files(directory):
    """
    Load all .npy files from a specified directory.

    Parameters:
    - directory (str): Path to the directory containing .npy files.

    Returns:
    - A dictionary with filenames as keys and loaded numpy arrays as values.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    data = {}

    for f in files:
        full_path = os.path.join(directory, f)
        data[f] = np.load(full_path)

    return data

# Usage
directory_path = './results/realdata/AMIS/'
loaded_data = load_all_npy_files(directory_path)

directory_path_alphaAMIS = './results/realdata/alphaAMIS/'
loaded_data_alphaAMIS = load_all_npy_files(directory_path_alphaAMIS)

dictionaries_data = [loaded_data, loaded_data_alphaAMIS]
results_baseline = {}
results_our_fixed_dof = {}
results_all = [results_baseline, results_our_fixed_dof]

set_keys_metrics, set_dofs, set_initialisation, set_nbruns, set_numsamples = set(), set(), set(), set(), set()

# First create dict
for i, data in enumerate(dictionaries_data):
    for key, _ in data.items():

        parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
        cleaned_parts = list(filter(lambda item: item != '', parts))

        # Cleaning
        if 'ESS' in cleaned_parts:
            idx = cleaned_parts.index('ESS')
            cleaned_parts[idx-1]+= 'ESS'
            cleaned_parts.remove('ESS')
        if 'alphaESS' in cleaned_parts:
            idx = cleaned_parts.index('alphaESS')
            cleaned_parts[idx-1]+= 'alphaESS'
            cleaned_parts.remove('alphaESS')
        if 'Z' in cleaned_parts:
            idx = cleaned_parts.index('Z')
            cleaned_parts[idx-1]+= 'Z'
            cleaned_parts.remove('Z')

        initialisation = cleaned_parts[0]
        numsamples = cleaned_parts[1]
        metric = cleaned_parts[-1] if i==1 else cleaned_parts[-2]
        dof = cleaned_parts[3]
        nbruns = cleaned_parts[5]

        set_keys_metrics.add(metric)
        set_dofs.add(dof)
        set_initialisation.add(initialisation)
        set_numsamples.add(numsamples)
        set_nbruns.add(nbruns)

results_final_baseline, results_final_ours= {}, {}

for metric in set_keys_metrics:
    results_final_baseline[metric], results_final_ours[metric] = {}, {}
    for initt in set_initialisation:
        results_final_baseline[metric][initt], results_final_ours[metric][initt] = {}, {}
        for dof in set_dofs:
            results_final_baseline[metric][initt][dof], results_final_ours[metric][initt][dof] = {}, {}
            for nbruns in set_nbruns:
                results_final_baseline[metric][initt][dof][nbruns], results_final_ours[metric][initt][dof][nbruns] = {}, {}
                for numsamples in set_numsamples:
                    results_final_baseline[metric][initt][dof][nbruns][numsamples], results_final_ours[metric][initt][dof][nbruns][numsamples] = [], []

for i, data in enumerate(dictionaries_data):
    for key, value in data.items():

        parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
        cleaned_parts = list(filter(lambda item: item != '', parts))

        # Cleaning
        if 'ESS' in cleaned_parts:
            idx = cleaned_parts.index('ESS')
            cleaned_parts[idx-1]+= 'ESS'
            cleaned_parts.remove('ESS')
        if 'alphaESS' in cleaned_parts:
            idx = cleaned_parts.index('alphaESS')
            cleaned_parts[idx-1]+= 'alphaESS'
            cleaned_parts.remove('alphaESS')
        if 'Z' in cleaned_parts:
            idx = cleaned_parts.index('Z')
            cleaned_parts[idx-1]+= 'Z'
            cleaned_parts.remove('Z')

        initialisation = cleaned_parts[0]
        numsamples = cleaned_parts[1]
        metric = cleaned_parts[-1] if i==1 else cleaned_parts[-2]
        dof = cleaned_parts[3]
        nbruns = cleaned_parts[5]

        if not value.shape == ():
            value = value[-1] # keeping only last value of iteration number

        if i == 0:
            results_final_baseline[metric][initialisation][dof][nbruns][numsamples].append(value.item())
        else: #(i == 1)
            results_final_ours[metric][initialisation][dof][nbruns][numsamples].append(value.item())


        # if key_metric not in results_all[i].keys():
        #     results_all[i][key_metric] = {}
        #
        #     if key_dof not in results_all[i][key_metric].keys():
        #         results_all[i][key_metric][key_dof] = {}
        #
        #         if key_numsamples not in results_all[i][key_metric][key_dof].keys():
        #             results_all[i][key_metric][key_dof][key_numsamples] = {}
        #
        #             if
        #                 results_all[i][key_metric][key_numsamples] = {}
        #         else:
        #             continue
        #     else:
        #         continue
        # else:
        #     if key_dof not in results_all[i][key_metric].keys():
        #         results_all[i][key_metric][key_dof] = {}
        #
        #         if key_numsamples not in results_all[i][key_metric][key_dof].keys():
        #             results_all[i][key_metric][key_numsamples][key_dof] = value[-1] # keeping only last value of iteration number
        #     else:
        #         continue



for dof in set_dofs:

    sorted_numsamples_baseline = sorted(list(map(int, results_final_baseline['meanalphaESS']['Laplaceinit'][dof]['20000'].keys() )))

    if dof == '5':
        truth_key = str(sorted_numsamples_baseline[-1])
        truth = results_final_baseline['meanalphaESS']['Laplaceinit'][dof]['1000'][truth_key][0]
        print(truth_key)

    sorted_numsamples_baseline = [x for x in sorted_numsamples_baseline if x in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]]

    sorted_numsamples_baseline_str = [str(num) for num in sorted_numsamples_baseline]

    # Step 2: Iterate through the sorted keys and get the corresponding values
    sorted_values_baseline = [ results_final_baseline['meanalphaESS']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_baseline_str]
    sorted_values_baseline_std = [ results_final_baseline['stdalphaESS']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_baseline_str]


    sorted_numsamples_ours = sorted(list(map(int, results_final_ours['meanalphaESS']['Laplaceinit'][dof]['20000'].keys() )))
    sorted_numsamples_ours = [x for x in sorted_numsamples_ours if x in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]]

    sorted_numsamples_str = [str(num) for num in sorted_numsamples_ours] # should be the same as baseline right ?

    # Step 2: Iterate through the sorted keys and get the corresponding values
    sorted_values_ours = [ results_final_ours['meanalphaESS']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_str]
    sorted_values_ours_std = [ results_final_ours['stdalphaESS']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_str]


    sorted_values_baseline, sorted_values_ours, sorted_values_baseline_std, sorted_values_ours_std = np.array(sorted_values_baseline), np.array(sorted_values_ours), np.array(sorted_values_baseline_std), np.array(sorted_values_ours_std)

    plt.plot(sorted_numsamples_baseline,  sorted_values_baseline, label='AMIS (dof={})'.format(dof))
    plt.plot(sorted_numsamples_ours,  sorted_values_ours, label='escort AMIS  (dof={})'.format(dof))

    plt.fill_between(sorted_numsamples_baseline, sorted_values_baseline - 1.96 * sorted_values_baseline_std, sorted_values_baseline + 1.96 * sorted_values_baseline_std, alpha=0.1)
    plt.fill_between(sorted_numsamples_ours, sorted_values_ours - 1.96 * sorted_values_ours_std, sorted_values_ours + 1.96 * sorted_values_ours_std, alpha=0.1)

plt.xlabel('Number of samples')
plt.ylabel('alphaESS')
# plt.axhline(y=truth, color='r', linestyle='solid', lw=2, label=r'AMIS with $10^5$ samples (dof=5)')
plt.tight_layout()
plt.legend()
# plt.savefig('./results/creatinine_alphaESS.pdf',bbox_inches='tight')
plt.show()


for dof in set_dofs:

    sorted_numsamples_baseline = sorted(list(map(int, results_final_baseline['meanZ']['Laplaceinit'][dof]['20000'].keys() )))

    # Best DOF it seems
    if dof == '5':
        truth_key = str(sorted_numsamples_baseline[-1])
        truth = results_final_baseline['meanZ']['Laplaceinit'][dof]['20000'][truth_key][0]

    sorted_numsamples_baseline = [x for x in sorted_numsamples_baseline if x in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]]

    sorted_numsamples_baseline_str = [str(num) for num in sorted_numsamples_baseline]

    # Step 2: Iterate through the sorted keys and get the corresponding values
    sorted_values_baseline = [ results_final_baseline['meanZ']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_baseline_str]
    sorted_values_baseline_std = [ results_final_baseline['stdZ']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_baseline_str]


    sorted_numsamples_ours = sorted(list(map(int, results_final_ours['meanZ']['Laplaceinit'][dof]['20000'].keys() )))
    sorted_numsamples_ours = [x for x in sorted_numsamples_ours if x in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]]

    sorted_numsamples_str = [str(num) for num in sorted_numsamples_ours] # should be the same as baseline right ?

    # Step 2: Iterate through the sorted keys and get the corresponding values
    sorted_values_ours = [ results_final_ours['meanZ']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_str]
    sorted_values_ours_std = [ results_final_ours['stdZ']['Laplaceinit'][dof]['20000'][key][0] for key in sorted_numsamples_str]


    sorted_values_baseline, sorted_values_ours, sorted_values_baseline_std, sorted_values_ours_std = np.array(sorted_values_baseline), np.array(sorted_values_ours), np.array(sorted_values_baseline_std), np.array(sorted_values_ours_std)

    plt.plot(sorted_numsamples_baseline,  sorted_values_baseline, label='AMIS (dof={})'.format(dof))
    plt.plot(sorted_numsamples_ours,  sorted_values_ours, label='escort AMIS  (dof={})'.format(dof))

    plt.fill_between(sorted_numsamples_baseline, sorted_values_baseline - 1.96 * sorted_values_baseline_std, sorted_values_baseline + 1.96 * sorted_values_baseline_std, alpha=0.1)
    plt.fill_between(sorted_numsamples_ours, sorted_values_ours - 1.96 * sorted_values_ours_std, sorted_values_ours + 1.96 * sorted_values_ours_std, alpha=0.1)

plt.axhline(y=truth, color='r', linestyle='solid', lw=2, label='Estimated true value (with $10^5$ samples)')
plt.xlabel('Number of samples')
plt.ylabel(r'$\hat{Z} value$')
plt.tight_layout()
plt.legend()
# plt.savefig('./results/creatinine_meanZ.pdf',bbox_inches='tight')
plt.show()

