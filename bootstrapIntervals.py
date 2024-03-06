import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import os

#load the data

conditionNumber = 5
# algName="AMIS/dof_5"
# algName="adaptive"

# algNameCollect = ["AMIS/dof_3", "AMIS/dof_5", "AMIS/dof_10", "escortAMIS/dof_3", "escortAMIS/dof_5", "escortAMIS/dof_10"]
# dofTargCollect = [5,50]

algNameCollect = ["AMIS/dof_3", "AMIS/dof_5","escortAMIS/dof_1", "escortAMIS/dof_2", "escortAMIS/dof_3", "escortAMIS/dof_5", "escortAMIS/dof_10"]
dofTargCollect = [2]

dimensionCollect = [2,4,8,16,32]

for algName in algNameCollect:
    for dofTarg in dofTargCollect:
        for dimension in dimensionCollect:

            folder = "./100Runs_dof"+str(dofTarg)+"_d"+str(dimension)+"_cond"+str(conditionNumber)+"/"+algName+"/allRuns"

            data = []
            for file in os.listdir(folder):
                if file.endswith("estimate_Z.txt"):
                    # print(file)
                    Z_estimate_from_run = np.loadtxt(folder+"/"+file, dtype="float")
                    # print(Z_estimate_from_run)
                    final_Z_estimate = Z_estimate_from_run[-1]
                    # print(final_Z_estimate)
                    data.append(final_Z_estimate)



            #construct confidence intervals 
            rng = np.random.default_rng()
            data = np.array(data)
            data = (data,)
            res = bootstrap(data, np.mean, confidence_level=0.9, random_state=rng)



            #writing the results
            folder_w = "./100Runs_dof"+str(dofTarg)+"_d"+str(dimension)+"_cond"+str(conditionNumber)+"/"+algName
            np.savetxt(folder_w+"/confidenceInterval.txt", res.confidence_interval)
