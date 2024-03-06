import matplotlib.pyplot as plt
import numpy as np
import os

plt.rc('text', usetex=True)

plt.rc('font', family='serif', size=22)
plt.rc('axes', labelsize=30, titlesize=20, labelpad=20)  # Adjusting axes parameters
plt.rc('xtick', labelsize=25)  # Adjusting xtick parameters
plt.rc('ytick', labelsize=25)  # Adjusting ytick parameters
plt.rc('legend', handlelength=2)  # e.g., shorter lines
plt.rc('lines', markersize=8)  # Replace 10 with your desired size
plt.rc('figure', figsize=(1.3 * 6.4, 1.3 * 4.8))  # For example, setting the figure size to 10 inches by 6 inches

conditionNumber = 5
dofTarg = 2
targetName = "dofTarget"+str(dofTarg)

dimensionCollect = [2,4,8,16,32]
d_collect = dimensionCollect

algCollect = [
    {'name':"AMIS/dof_3", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"AMIS/dof_5", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    # {'name':"AMIS/dof_10", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"escortAMIS/dof_1", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"escortAMIS/dof_2", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"escortAMIS/dof_3", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"escortAMIS/dof_5", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    # {'name':"escortAMIS/dof_10", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]},
    {'name':"adaptive", 'MSE_Z':[], 'alphaESS_m':[], 'alphaESS_std':[], 'conf_lb':[], 'conf_ub':[]}
    ]


#collecting the results from experiments

true_Z = []

for dimension in dimensionCollect:
    folder = "./100Runs_dof"+str(dofTarg)+"_d"+str(dimension)+"_cond"+str(conditionNumber)
    Z = np.loadtxt(folder+"/true_Z", dtype="float")
    true_Z.append(Z)


for alg in algCollect:

    for dimension in dimensionCollect:

        folder = "./100Runs_dof"+str(dofTarg)+"_d"+str(dimension)+"_cond"+str(conditionNumber)+"/"+alg['name']
        
        mse_z = np.loadtxt(folder+"/MSE_Z_m.txt", dtype="float")
        alg['MSE_Z'].append(mse_z[-1])

        alpha_ess_m = np.loadtxt(folder+"/alphaESS_m.txt", dtype="float")
        alg['alphaESS_m'].append(alpha_ess_m[-1])
        alpha_ess_std = np.loadtxt(folder+"/alphaESS_std.txt", dtype="float")
        alg['alphaESS_std'].append(alpha_ess_std[-1])

        confidence_bounds = np.loadtxt(folder+"/confidenceInterval.txt", dtype="float")
        alg['conf_lb'].append(confidence_bounds[0])
        alg['conf_ub'].append(confidence_bounds[1])
    
    alg['MSE_Z'] = np.array(alg['MSE_Z'])
    alg['alphaESS_m'] = np.array(alg['alphaESS_m'])
    alg['alphaESS_std'] = np.array(alg['alphaESS_std'])
    alg['conf_lb'] = np.array(alg['conf_lb'])
    alg['conf_ub'] = np.array(alg['conf_ub'])



# gathering the experiments

index_AMIS_dof3 = 0
index_AMIS_dof5 = 1
index_escortAMIS_dof1 = 2
index_escortAMIS_dof2 = 3
index_escortAMIS_dof3 = 4
index_escortAMIS_dof5 = 5
index_adaptive = 6

# index_AMIS_dof3 = 0
# index_AMIS_dof5 = 1
# index_AMIS_dof10 = 2
# index_escortAMIS_dof3 = 3
# index_escortAMIS_dof5 = 4
# index_escortAMIS_dof10 = 5
# index_adaptive = 6

MSE_Z_AMIS_dof3 = algCollect[index_AMIS_dof3]['MSE_Z']
MSE_Z_AMIS_dof5 = algCollect[index_AMIS_dof5]['MSE_Z']
# MSE_Z_AMIS_dof10 = algCollect[index_AMIS_dof10]['MSE_Z']
MSE_Z_escortAMIS_dof1 = algCollect[index_escortAMIS_dof1]['MSE_Z']
MSE_Z_escortAMIS_dof2 = algCollect[index_escortAMIS_dof2]['MSE_Z']
MSE_Z_escortAMIS_dof3 = algCollect[index_escortAMIS_dof3]['MSE_Z']
MSE_Z_escortAMIS_dof5 = algCollect[index_escortAMIS_dof5]['MSE_Z']
# MSE_Z_escortAMIS_dof10 = algCollect[index_escortAMIS_dof10]['MSE_Z']
MSE_Z_adaptive = algCollect[index_adaptive]['MSE_Z']

alphaESS_mean_AMIS_dof3 = algCollect[index_AMIS_dof3]['alphaESS_m']
alphaESS_mean_AMIS_dof5 = algCollect[index_AMIS_dof5]['alphaESS_m']
# alphaESS_mean_AMIS_dof10 = algCollect[index_AMIS_dof10]['alphaESS_m']
alphaESS_mean_escortAMIS_dof1 = algCollect[index_escortAMIS_dof1]['alphaESS_m']
alphaESS_mean_escortAMIS_dof2 = algCollect[index_escortAMIS_dof2]['alphaESS_m']
alphaESS_mean_escortAMIS_dof3 = algCollect[index_escortAMIS_dof3]['alphaESS_m']
alphaESS_mean_escortAMIS_dof5 = algCollect[index_escortAMIS_dof5]['alphaESS_m']
# alphaESS_mean_escortAMIS_dof10 = algCollect[index_escortAMIS_dof10]['alphaESS_m']
alphaESS_mean_adaptive = algCollect[index_adaptive]['alphaESS_m']

alphaESS_std_AMIS_dof3 = algCollect[index_AMIS_dof3]['alphaESS_std']
alphaESS_std_AMIS_dof5 = algCollect[index_AMIS_dof5]['alphaESS_std']
# alphaESS_std_AMIS_dof10 = algCollect[index_AMIS_dof10]['alphaESS_std']
alphaESS_std_escortAMIS_dof1 = algCollect[index_escortAMIS_dof1]['alphaESS_std']
alphaESS_std_escortAMIS_dof2 = algCollect[index_escortAMIS_dof2]['alphaESS_std']
alphaESS_std_escortAMIS_dof3 = algCollect[index_escortAMIS_dof3]['alphaESS_std']
alphaESS_std_escortAMIS_dof5 = algCollect[index_escortAMIS_dof5]['alphaESS_std']
# alphaESS_std_escortAMIS_dof10 = algCollect[index_escortAMIS_dof10]['alphaESS_std']
alphaESS_std_adaptive = algCollect[index_adaptive]['alphaESS_std']

conf_lb_AMIS_dof3 = algCollect[index_AMIS_dof3]['conf_lb']
conf_lb_AMIS_dof5 = algCollect[index_AMIS_dof5]['conf_lb']
# conf_lb_AMIS_dof10 = algCollect[index_AMIS_dof10]['conf_lb']
conf_lb_escortAMIS_dof1 = algCollect[index_escortAMIS_dof1]['conf_lb']
conf_lb_escortAMIS_dof2 = algCollect[index_escortAMIS_dof2]['conf_lb']
conf_lb_escortAMIS_dof3 = algCollect[index_escortAMIS_dof3]['conf_lb']
conf_lb_escortAMIS_dof5 = algCollect[index_escortAMIS_dof5]['conf_lb']
# conf_lb_escortAMIS_dof10 = algCollect[index_escortAMIS_dof10]['conf_lb']
conf_lb_adaptive = algCollect[index_adaptive]['conf_lb']

conf_ub_AMIS_dof3 = algCollect[index_AMIS_dof3]['conf_ub']
conf_ub_AMIS_dof5 = algCollect[index_AMIS_dof5]['conf_ub']
# conf_ub_AMIS_dof10 = algCollect[index_AMIS_dof10]['conf_ub']
conf_ub_escortAMIS_dof1 = algCollect[index_escortAMIS_dof1]['conf_ub']
conf_ub_escortAMIS_dof2 = algCollect[index_escortAMIS_dof2]['conf_ub']
conf_ub_escortAMIS_dof3 = algCollect[index_escortAMIS_dof3]['conf_ub']
conf_ub_escortAMIS_dof5 = algCollect[index_escortAMIS_dof5]['conf_ub']
# conf_ub_escortAMIS_dof10 = algCollect[index_escortAMIS_dof10]['conf_ub']
conf_ub_adaptive = algCollect[index_adaptive]['conf_ub']



# MSE plotting

plt.figure()
plt.semilogy()

plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof1) / true_Z, label="escort AMIS, $\\nu=1$", marker="o")
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof2) / true_Z, label="escort AMIS, $\\nu=2$", marker="|")
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof3) / true_Z, label="escort AMIS, $\\nu=3$", marker="v")
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof5) / true_Z, label="escort AMIS, $\\nu=5$", marker="p")
# plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof10) / true_Z, label="escort AMIS, $\\nu=10$", marker="*")
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof3) / true_Z, label="AMIS, $\\nu=3$", linestyle="dashed", marker="v")
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof5) / true_Z, label="AMIS, $\\nu=5$", linestyle="dashed", marker="p")
# plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof10) / true_Z, label="AMIS, $\\nu=10$", linestyle="dashed", marker="*")
plt.plot(d_collect, np.sqrt(MSE_Z_adaptive) / true_Z, label="adaptive escort AMIS", marker="x")

plt.legend()

plt.xlabel("Dimension $d$")
plt.ylabel("$\\sqrt{MSE}\\, /\\, Z_{\\pi}$")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

plt.grid(which='both')  # showing both major and minor grid lines


plt.savefig("MSE_Z_"+targetName+".pdf",bbox_inches="tight")



# alpha ESS plotting

plt.figure()
plt.semilogy()

plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof1 - alphaESS_std_escortAMIS_dof1, alphaESS_mean_escortAMIS_dof1 + alphaESS_std_escortAMIS_dof1, alpha=0.2)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof2 - alphaESS_std_escortAMIS_dof2, alphaESS_mean_escortAMIS_dof2 + alphaESS_std_escortAMIS_dof2, alpha=0.2)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof3 - alphaESS_std_escortAMIS_dof3, alphaESS_mean_escortAMIS_dof3 + alphaESS_std_escortAMIS_dof3, alpha=0.2)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof5 - alphaESS_std_escortAMIS_dof5, alphaESS_mean_escortAMIS_dof5 + alphaESS_std_escortAMIS_dof5, alpha=0.2)
# plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof10 - alphaESS_std_escortAMIS_dof10, alphaESS_mean_escortAMIS_dof10 + alphaESS_std_escortAMIS_dof10, alpha=0.2)

plt.fill_between(d_collect, alphaESS_mean_AMIS_dof3 - alphaESS_std_AMIS_dof3, alphaESS_mean_AMIS_dof3 + alphaESS_std_AMIS_dof3, alpha=0.2)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof5 - alphaESS_std_AMIS_dof5, alphaESS_mean_AMIS_dof5 + alphaESS_std_AMIS_dof5, alpha=0.2)
# plt.fill_between(d_collect, alphaESS_mean_AMIS_dof10 - alphaESS_std_AMIS_dof10, alphaESS_mean_AMIS_dof10 + alphaESS_std_AMIS_dof10, alpha=0.2)

plt.fill_between(d_collect, alphaESS_mean_adaptive - alphaESS_std_adaptive, alphaESS_mean_adaptive + alphaESS_std_adaptive, alpha=0.2)


plt.plot(d_collect, alphaESS_mean_escortAMIS_dof1, label="escort AMIS, $\\nu=1$", marker="o")
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof2, label="escort AMIS, $\\nu=2$", marker="|")
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof3, label="escort AMIS, $\\nu=3$", marker="v")
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof5, label="escort AMIS, $\\nu=5$", marker="p")
# plt.plot(d_collect, alphaESS_mean_escortAMIS_dof10, label="escort AMIS, $\\nu=1$0", marker="*")
plt.plot(d_collect, alphaESS_mean_AMIS_dof3, label="AMIS, $\\nu=3$", linestyle="dashed", marker="v")
plt.plot(d_collect, alphaESS_mean_AMIS_dof5, label="AMIS, $\\nu=5$", linestyle="dashed", marker="p")
# plt.plot(d_collect, alphaESS_mean_AMIS_dof10, label="AMIS, $\\nu=10$", linestyle="dashed", marker="*")
plt.plot(d_collect, alphaESS_mean_adaptive, label="adaptive escort AMIS", marker="x")

plt.legend()

plt.xlabel("Dimension $d$")
plt.ylabel("$\\alpha$-ESS")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

plt.grid(which='both')  # showing both major and minor grid lines


plt.savefig("alphaESS_"+targetName+".pdf",bbox_inches="tight")


# confidence interval plotting

plt.figure()
plt.semilogy()

plt.fill_between(d_collect, conf_lb_escortAMIS_dof1 / true_Z, conf_ub_escortAMIS_dof1 / true_Z, alpha=0.5, label="escort AMIS, $\\nu=1$")
plt.fill_between(d_collect, conf_lb_escortAMIS_dof2 / true_Z, conf_ub_escortAMIS_dof2 / true_Z, alpha=0.5, label="escort AMIS, $\\nu=2$")
plt.fill_between(d_collect, conf_lb_escortAMIS_dof3 / true_Z, conf_ub_escortAMIS_dof3 / true_Z, alpha=0.5, label="escort AMIS, $\\nu=3$")
plt.fill_between(d_collect, conf_lb_escortAMIS_dof5 / true_Z, conf_ub_escortAMIS_dof5 / true_Z, alpha=0.5, label="escort AMIS, $\\nu=5$")
# plt.fill_between(d_collect, conf_lb_escortAMIS_dof10 / true_Z, conf_ub_escortAMIS_dof10 / true_Z, alpha=0.5, label="escort AMIS, $\\nu=10$")

plt.fill_between(d_collect, conf_lb_AMIS_dof3 / true_Z, conf_ub_AMIS_dof3 / true_Z, alpha=0.5, label="AMIS, $\\nu=3$")
plt.fill_between(d_collect, conf_lb_AMIS_dof5 / true_Z, conf_ub_AMIS_dof5 / true_Z, alpha=0.5, label="AMIS, $\\nu=5$")
# plt.fill_between(d_collect, conf_lb_AMIS_dof10 / true_Z, conf_ub_AMIS_dof10 / true_Z, alpha=0.5, label="AMIS, $\\nu=10$")

plt.fill_between(d_collect, conf_lb_adaptive / true_Z, conf_ub_adaptive / true_Z, alpha=0.5, label="AHTIS")


plt.legend(loc='lower left')

plt.xlabel("Dimension $d$")
plt.ylabel(r" Interval: $[\widehat{L}/Z_{\pi}, \widehat{U}/Z_{\pi}$ ] ")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

plt.grid(which='both')  # showing both major and minor grid lines


plt.savefig("confidenceIntervals_"+targetName+".pdf",bbox_inches="tight")







