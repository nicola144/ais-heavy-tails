from synthetic_targets import *
from amis_algorithms import *
import os
from functools import partial
from tqdm import tqdm


def run_adaptiveAMISrealdataset(nb_runs, n_iterations, dof_proposal, M, d, alg, sigmaSq_init, log_pi_tilde):

    all_est_Z = np.empty((nb_runs,n_iterations))
    ESS = np.empty((nb_runs, n_iterations))
    alphaESS = np.empty((nb_runs, n_iterations))

    dof = np.empty((nb_runs, n_iterations))

    shape_initial = sigmaSq_init * np.eye(d)

    for i in tqdm(range(nb_runs)):

        mu_initial = np.random.uniform(-5, 5, d)

        all_estimate_Z, all_alphaESS, all_ESS, all_dof = alg(mu_initial, shape_initial, n_iterations, log_pi_tilde,
                                                             dof_proposal, M, d)


        all_est_Z[i, :] = all_estimate_Z # only last iteration
        ESS[i, :] = all_ESS
        alphaESS[i, :] = all_alphaESS
        dof[i, :] = all_dof

    return all_est_Z, ESS, alphaESS, dof


def run_AMIS_real_dataset(nb_runs, n_iterations, dof_proposal, M, d, alg, log_pi_tilde, mu_initial=None, shape_initial=None, sigmaSq_init=None):
    all_est_Z = np.empty((nb_runs,n_iterations))
    ESS = np.empty((nb_runs, n_iterations))
    alphaESS = np.empty((nb_runs, n_iterations))

    shape_initial = sigmaSq_init * np.identity(d)

    assert np.all(np.linalg.eigvals(shape_initial) > 0)

    for i in tqdm(range(nb_runs)):

        mu_initial = np.random.uniform(-5,5,d)

        all_estimate_Z, all_alphaESS, all_ESS = alg(mu_initial=mu_initial, shape_initial=shape_initial, n_iterations=n_iterations, log_pi_tilde=log_pi_tilde, dof_proposal=dof_proposal,
                                                    M=M, D=d)


        all_est_Z[i, :] = all_estimate_Z # only last iteration
        ESS[i, :] = all_ESS
        alphaESS[i, :] = all_alphaESS

    # mean_Z = all_est_Z.mean(0)
    # std_Z = all_est_Z.std(0)
    # mean_ESS = ESS.mean(0)
    # mean_alphaESS = alphaESS.mean(0)
    #
    # std_ESS = ESS.std(0)
    # std_alphaESS = alphaESS.std(0)

    return all_est_Z, ESS, alphaESS

def run_AMIS(nb_runs, n_iterations, dof_proposal, M, d, alg, sigmaSq_init, log_pi_tilde, Z_target):
    MSE_Z = np.empty((nb_runs, n_iterations))
    ESS = np.empty((nb_runs, n_iterations))
    alphaESS = np.empty((nb_runs, n_iterations))

    # MSE_Z = np.zeros(n_iterations)
    # ESS = np.zeros(n_iterations)
    # alphaESS = np.zeros(n_iterations)

    for i in range(nb_runs):

        mu_initial = np.random.uniform(-5, 5, d)
        shape_initial = sigmaSq_init * np.eye(d)

        all_estimate_Z, all_alphaESS, all_ESS = alg(mu_initial=mu_initial, shape_initial=shape_initial, n_iterations=n_iterations, log_pi_tilde=log_pi_tilde, dof_proposal=dof_proposal,
                                                    M=M, D=d)

        SE_Z = np.empty(n_iterations)
        for n in range(n_iterations):
            SE_Z[n] = (all_estimate_Z[n] - Z_target) ** 2

        # MSE_Z += (1/nb_runs)*SE_Z
        # ESS += (1/nb_runs)*all_ESS
        # alphaESS += (1/nb_runs)*all_alphaESS

        MSE_Z[i, :] = SE_Z
        ESS[i, :] = all_ESS
        alphaESS[i, :] = all_alphaESS

    mean_MSE_Z = MSE_Z.mean(0)
    mean_ESS = ESS.mean(0)
    mean_alphaESS = alphaESS.mean(0)

    std_MSE_Z = MSE_Z.std(0)
    std_ESS = ESS.std(0)
    std_alphaESS = alphaESS.std(0)

    return mean_MSE_Z, mean_ESS, mean_alphaESS, std_MSE_Z, std_ESS, std_alphaESS




def main():
    d_collect = [2]
    dof_targ = 3
    cond_number = 2
    dof_proposal_collect = [1]

    sigmaSq_init = 5
    M = 20000
    nb_runs = 50
    nb_iterations = 20

    for d in d_collect:

        loc_targ = np.random.uniform(-1, 1, d)
        shape_targ = matrix_condition(d, cond_number)
        inv_shape_targ = np.linalg.inv(shape_targ)
        Z_target = normalization_Student(d, dof_targ, shape_targ)
        # log_pi_tilde = lambda x: unnormalized_logpdf_student(x, dof, loc, inv_shape)
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

        print('Target dof', dof_targ)

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

            np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_m.txt",
                       mean_MSE_Z_escortAMIS)
            np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_m.txt",
                       mean_ESS_escortAMIS)
            np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_alphaESS_m.txt",
                       mean_alphaESS_escortAMIS)
            np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_MSE_Z_std.txt",
                       std_MSE_Z_escortAMIS)
            np.savetxt('./results/' + target_name + "/escortAMIS/dof" + str(dof_prop) + "_ESS_std.txt",
                       std_ESS_escortAMIS)
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


if __name__ == "__main__":
    main()

