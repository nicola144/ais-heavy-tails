from synthetic_targets import *
from amis_algorithms import *
import os
from functools import partial

d_collect = [2]
dof_proposal_collect = [1]

M = 20000
nb_runs = 50
nb_iterations = 20


log_pi_tilde = partial(unnormalized_logpdf_Student, dof=dof_targ, loc=loc_targ, inv_shape=inv_shape_targ)
target_name = "dof" + str(dof_targ) + "_d" + str(d)

if not os.path.exists('./results/' + target_name):
    os.mkdir('./results/' + target_name)

if not os.path.exists('./results/' + target_name + '/AMIS'):
    os.mkdir('./results/' + target_name + '/AMIS')

if not os.path.exists('./results/' + target_name + '/escortAMIS'):
    os.mkdir('./results/' + target_name + '/escortAMIS')

fig = plt.figure()
plt.semilogy()
iterations = range(nb_iterations)


for dof_prop in dof_proposal_collect:

    # if dof_prop > 2:
    #     mean_MSE_Z_AMIS, mean_ESS_AMIS, mean_alphaESS_AMIS, std_MSE_Z_AMIS, std_ESS_AMIS, std_alphaESS_AMIS = run_AMIS(
    #         nb_runs, nb_iterations, dof_prop, M, d, AMIS_student_fixed_dof, sigmaSq_init, log_pi_tilde, Z_target)
    #
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_MSE_Z_m.txt", mean_MSE_Z_AMIS)
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_ESS_m.txt", mean_ESS_AMIS)
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_alphaESS_m.txt", mean_alphaESS_AMIS)
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_MSE_Z_std.txt", std_MSE_Z_AMIS)
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_ESS_std.txt", std_ESS_AMIS)
    #     np.savetxt('./results/' + target_name + "/AMIS/dof" + str(dof_prop) + "_alphaESS_std.txt", std_alphaESS_AMIS)
    #
    #     plt.plot(iterations, mean_MSE_Z_AMIS, label="AMIS, dof=" + str(dof_prop))
    #
    # mean_MSE_Z_escortAMIS, mean_ESS_escortAMIS, mean_alphaESS_escortAMIS, std_MSE_Z_escortAMIS, std_ESS_escortAMIS, std_alphaESS_escortAMIS = run_AMIS(
    #     nb_runs, nb_iterations, dof_prop, M, d, alpha_AMIS_fixed_dof, sigmaSq_init, log_pi_tilde, Z_target)
    #
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_m.txt", mean_MSE_Z_escortAMIS)
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_m.txt", mean_ESS_escortAMIS)
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_alphaESS_m.txt", mean_alphaESS_escortAMIS)
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_std.txt", std_MSE_Z_escortAMIS)
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_std.txt", std_ESS_escortAMIS)
    # np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_alphaESS_std.txt",
    #            std_alphaESS_escortAMIS)
    #
    # plt.plot(iterations, mean_MSE_Z_escortAMIS, label="escort AMIS, dof=" + str(dof_prop))


    mean_MSE_Z_escortAMIS, mean_ESS_escortAMIS, mean_alphaESS_escortAMIS, std_MSE_Z_escortAMIS, std_ESS_escortAMIS, std_alphaESS_escortAMIS = run_AMIS(
        nb_runs, nb_iterations, dof_prop, M, d, alpha_AMIS_adapted_dof, sigmaSq_init, log_pi_tilde, Z_target)

    sys.exit(0)

    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_m.txt", mean_MSE_Z_escortAMIS)
    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_m.txt", mean_ESS_escortAMIS)
    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_alphaESS_m.txt", mean_alphaESS_escortAMIS)
    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_std.txt", std_MSE_Z_escortAMIS)
    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_std.txt", std_ESS_escortAMIS)
    np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_alphaESS_std.txt",
               std_alphaESS_escortAMIS)

    plt.plot(iterations, mean_MSE_Z_escortAMIS, label="escort AMIS, dof=" + str(dof_prop))


plt.legend()
plt.savefig("./" + target_name + "/MSE_Z.pdf")

# d = 5
# dof_targ = 5

# condition_nb = 10
# loc_targ = np.random.uniform(-1,1,d)
# shape_targ = matrix_condition(d,condition_nb)
# inv_shape_targ = np.linalg.inv(shape_targ)
# Z_target = normalization_Student(d, dof_targ, shape_targ)

# log_pi_tilde = partial(unnormalized_logpdf_Student, dof=dof_targ, loc=loc_targ, inv_shape=inv_shape_targ)
# target_name = "dof"+str(dof_targ)+"_d"+str(d)


# M = 10000
# dof_AMIS = max(2.5, dof_targ)
# dof_eAMIS = dof_targ
# n_iterations = 20
# nb_runs = 20
# sigmaSq_init = 10

# AMIS = AMIS_student_fixed_dof
# escortAMIS = alpha_AMIS_fixed_dof

# MSE_Z_AMIS, ESS_AMIS, alphaESS_AMIS, std_MSE_Z_AMIS, std_ESS_AMIS, std_alphaESS_AMIS = run_AMIS(nb_runs, n_iterations, dof_AMIS, M, d, AMIS, sigmaSq_init, log_pi_tilde, Z_target)
# MSE_Z_escortAMIS, ESS_escortAMIS, alphaESS_escortAMIS, std_MSE_Z_escortAMIS, std_ESS_escortAMIS, std_alphaESS_escortAMIS = run_AMIS(nb_runs, n_iterations, dof_eAMIS, M, d, escortAMIS, sigmaSq_init, log_pi_tilde, Z_target)


# iterations = range(n_iterations)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(iterations, MSE_Z_AMIS, label="AMIS")
# ax1.plot(iterations, MSE_Z_escortAMIS, label="escort AMIS")
# ax1.set_yscale("log")
# ax2.plot(iterations, ESS_AMIS, label="AMIS")
# ax2.plot(iterations, ESS_escortAMIS, label="escort AMIS")
# ax1.legend()
# ax2.legend()
# plt.show()


