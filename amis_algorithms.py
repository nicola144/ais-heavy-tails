import numpy as np

from utils import *

from tqdm import tqdm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t, random_correlation
# from emukit.core.loop import UserFunctionWrapper
# from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
# from emukit.core.loop import UserFunctionResult
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, MaxValueEntropySearch, NegativeLowerConfidenceBound
import GPy
from emukit.core.initial_designs.latin_design import LatinDesign
from functools import partial

def AMIS_student_fixed_dof(mu_initial,shape_initial, n_iterations, log_pi_tilde, dof_proposal, M, D):
    
    alpha = 1 + 2 / (dof_proposal + D) #to be used for computation of the alpha-ESS
    
    all_samples = np.empty((n_iterations,M,D))
    evaluations_target_logpdf = np.empty((n_iterations,M))
    mixture_denominator_evaluations = np.empty((n_iterations,M))
    proposals_over_iterations = []

    all_estimate_Z = np.empty(n_iterations)
    all_ESS = np.empty(n_iterations)
    all_alphaESS = np.empty(n_iterations)

    mu_current = mu_initial
    shape_current = shape_initial

    # if D == 2:
    #     nb_points = 100
    #     X = np.linspace(-10, 10, nb_points)
    #     Y = np.linspace(-10, 10, nb_points)


    #     pdf = np.zeros((nb_points, nb_points))
    #     for i in range(nb_points):
    #         for j in range(nb_points):
    #             pdf[i,j] = np.exp(log_pi_tilde([X[i], Y[j]]))

    #     # plot
    #     pdf = (1 / sum(pdf)) * pdf
    #     fig, ax = plt.subplots()
    #     ax.contour(X, Y, pdf)
    
    # Iterations
    for t in range(n_iterations):
        
        current_proposal = multivariate_t(loc=mu_current, shape=shape_current, df=dof_proposal)
        proposals_over_iterations.append(current_proposal)

        # Draw M samples from current proposal
        samples_current = current_proposal.rvs(size=M)
        all_samples[t,:] = samples_current # this adds to the existing list of samples, does not override

        

        # Numerator
        for m in range(M):
            evaluations_target_logpdf[t,m] = log_pi_tilde(samples_current[m,:])#log_pi_tilde may not be broasdact-compatible

        log_numerator = evaluations_target_logpdf[:t+1,:] # t+1 since including current ones !

        
        # Computing the DM weights in the denominator
        current_only_log_denominator = current_proposal.logpdf(samples_current)
        mixture_denominator_evaluations[t,:] = np.exp(current_only_log_denominator)
        
        for tau in range(t):
            #this loop is only entered if t>0
            mixture_denominator_evaluations[tau,:] += np.exp(current_proposal.logpdf(all_samples[tau,:]))
            mixture_denominator_evaluations[t,:] += np.exp(proposals_over_iterations[tau].logpdf(samples_current))
        
        log_denominator = - np.log(t+1) +  np.log(mixture_denominator_evaluations[:t+1,:])

        assert log_numerator.shape == log_denominator.shape

        updated_logweights = log_numerator - log_denominator
        logsumexp_logweights = logsumexp(updated_logweights)
        updated_normalized_logweights = updated_logweights - logsumexp_logweights
        #print(np.exp(updated_normalized_logweights))
        
        ### Estimate of the normalization constant
        estimate_Z = np.exp(logsumexp_logweights) / (M*(t+1))
        all_estimate_Z[t] = estimate_Z

        
        ### metrics 
        current_only_logweights = evaluations_target_logpdf[t,:] - current_only_log_denominator
        current_only_normalized_logweights = current_only_logweights - logsumexp(current_only_logweights)

        #print(np.exp(current_only_normalized_logweights))
        #print(max(np.exp(current_only_normalized_logweights)))
        current_only_alphaESS = np.exp(logsumexp(alpha*current_only_normalized_logweights))**(1 / (1-alpha))
        current_only_ESS = 1 / np.exp(logsumexp(2*current_only_normalized_logweights))
        
        all_alphaESS[t] = (1/M)*current_only_alphaESS
        all_ESS[t] = (1/M)*current_only_ESS
        
        ### Update proposal
        
        samples_up_to_now = all_samples[:t+1,:,:]

        # first_moment = np.zeros(D)
        # secnd_moment = np.zeros((D,D))
        # for tau in range(t+1):
        #     for m in range(M):
        #         w = np.exp(updated_normalized_logweights[tau, m])
        #         first_moment += w * samples_up_to_now[tau, m, :]
        #         secnd_moment += w * (samples_up_to_now[tau, m, :].reshape(-1, 1) @ samples_up_to_now[tau, m, :].reshape(1, -1))

        # Standard moment matching
        W = np.exp(updated_normalized_logweights)
        mu_current = np.einsum('tmd,tm->d', samples_up_to_now, W)
        secnd_moment = np.einsum('tm, tmd, tme -> de', W, samples_up_to_now, samples_up_to_now)
        shape_current = ((dof_proposal - 2) / dof_proposal) * ( secnd_moment - mu_current.reshape(-1, 1) @ mu_current.reshape(1, -1) )

    return all_estimate_Z, all_alphaESS, all_ESS





def alpha_AMIS_fixed_dof(mu_initial,shape_initial, n_iterations, log_pi_tilde, dof_proposal, M, D):
    
    alpha = 1 + 2 / (dof_proposal + D) #to be used for computation of the alpha-ESS
    
    all_samples = np.empty((n_iterations,M,D))
    evaluations_target_logpdf = np.empty((n_iterations,M))
    mixture_denominator_evaluations = np.empty((n_iterations,M))
    proposals_over_iterations = []
    # Statistics to track
    all_estimate_Z = np.empty(n_iterations)
    all_ESS = np.empty(n_iterations)
    all_alphaESS = np.empty(n_iterations)

    mu_current = mu_initial
    shape_current = shape_initial
    
    # Iterations
    for t in range(n_iterations):
        current_proposal = multivariate_t(loc=mu_current, shape=shape_current, df=dof_proposal)
        proposals_over_iterations.append(current_proposal)

        # Draw M samples from current proposal
        samples_current = current_proposal.rvs(size=M)
        all_samples[t,:] = samples_current # this adds to the existing list of samples, does not override

        # Numerator
        for m in range(M):
            evaluations_target_logpdf[t,m] = log_pi_tilde(samples_current[m,:])# log_pi_tilde may not be broasdact-compatible

        log_numerator = evaluations_target_logpdf[:t+1,:] # t+1 since including current ones !

        
        # Computing the DM weights in the denominator
        current_only_log_denominator = current_proposal.logpdf(samples_current)
        mixture_denominator_evaluations[t,:] = np.exp(current_only_log_denominator)
        
        for tau in range(t):
            #this loop is only entered if t>0
            mixture_denominator_evaluations[tau,:] += np.exp(current_proposal.logpdf(all_samples[tau,:]))
            mixture_denominator_evaluations[t,:] += np.exp(proposals_over_iterations[tau].logpdf(samples_current))
        
        log_denominator = - np.log(t+1) +  np.log(mixture_denominator_evaluations[:t+1,:])

        assert log_numerator.shape == log_denominator.shape

        updated_logweights = log_numerator - log_denominator
        logsumexp_logweights = logsumexp(updated_logweights)
        updated_normalized_logweights = updated_logweights - logsumexp_logweights
        
        
        ### Estimate of the normalization constant
        estimate_Z = np.exp(logsumexp_logweights) / (M*(t+1)) # weights need to have only pi_tilde in the numerator (no escort) 
        all_estimate_Z[t] = estimate_Z

        ### metrics 
        current_only_logweights = evaluations_target_logpdf[t,:] - current_only_log_denominator 
        current_only_normalized_logweights = current_only_logweights - logsumexp(current_only_logweights) 

        # weights need to have only pi_tilde in the numerator (no escort)
        current_only_alphaESS = np.exp(logsumexp(alpha*current_only_normalized_logweights))**(1 / (1-alpha))
        current_only_ESS = 1 / np.exp(logsumexp(2*current_only_normalized_logweights))
        
        all_alphaESS[t] = (1/M)*current_only_alphaESS
        all_ESS[t] = (1/M)*current_only_ESS
        
        ### Update proposal
        
        samples_up_to_now = all_samples[:t+1,:,:]

        # weights need to have the escort of pi_tilde in the numerator
        updated_escort_logweights = alpha*log_numerator - log_denominator
        updated_normalized_escort_logweights = updated_escort_logweights - logsumexp(updated_escort_logweights)

        # first_moment = np.zeros(D)
        # secnd_moment = np.zeros((D,D))

        # for tau in range(t+1):
        #     for m in range(M):
        #         w = np.exp(updated_normalized_escort_logweights[tau, m])
        #         first_moment += w * samples_up_to_now[tau, m, :]
        #         secnd_moment += w * (samples_up_to_now[tau, m, :].reshape(-1, 1) @ samples_up_to_now[tau, m, :].reshape(1, -1))

        # mu_current = first_moment
        # shape_current =  secnd_moment - (mu_current.reshape(-1, 1) @ mu_current.reshape(1, -1))

        W = np.exp(updated_normalized_escort_logweights)
        mu_current = np.einsum('tmd,tm->d', samples_up_to_now, W)
        secnd_moment =  np.einsum('tm, tmd, tme -> de', W, samples_up_to_now, samples_up_to_now)
        shape_current = secnd_moment - (mu_current.reshape(-1, 1) @ mu_current.reshape(1, -1))

    return all_estimate_Z, all_alphaESS, all_ESS


def alpha_AMIS_adapted_dof(mu_initial,shape_initial, n_iterations, log_pi_tilde, dof_proposal, M, D):

    

    all_samples = np.empty((n_iterations, M, D))
    evaluations_target_logpdf = np.empty((n_iterations, M))
    mixture_denominator_evaluations = np.empty((n_iterations, M))
    proposals_over_iterations = []

    # Statistics to track
    all_estimate_Z = np.empty(n_iterations)
    all_ESS = np.empty(n_iterations)
    all_alphaESS = np.empty(n_iterations)
    all_dof = np.empty(n_iterations)

    # Map of available acquisition functions
    acquisition_functions = {
        'EI': ExpectedImprovement,
        'LCB': NegativeLowerConfidenceBound,
        'MES': MaxValueEntropySearch
    }

          

    gpy_model = None
    emukit_model = None

    # Range DOF from 1 to 10
    dof_proposal_space = ParameterSpace([ContinuousParameter('dof_proposal', 1, 10)])
    num_initial_dof_points = 5


    observed_dof = np.array([])
    observed_ess = np.array([])

    mu_current = mu_initial
    shape_current = shape_initial

    # Initial points
    latin_design = LatinDesign(dof_proposal_space)
    initial_dof_points = latin_design.get_samples(num_initial_dof_points)

    # Iterations
    for t in range(n_iterations):
        
        alpha = 1 + 2 / (dof_proposal + D)
        

        current_proposal = multivariate_t(loc=mu_current, shape=shape_current, df=dof_proposal)
        proposals_over_iterations.append(current_proposal)

        # Draw M samples from current proposal
        samples_current = current_proposal.rvs(size=M)
        all_samples[t, :] = samples_current  # this adds to the existing list of samples, does not override

        # Numerator
        for m in range(M):
            evaluations_target_logpdf[t, m] = log_pi_tilde(
                samples_current[m, :])  # log_pi_tilde may not be broasdact-compatible

        log_numerator = evaluations_target_logpdf[:t + 1, :]  # t+1 since including current ones !

        # Computing the DM weights in the denominator
        current_only_log_denominator = current_proposal.logpdf(samples_current)
        mixture_denominator_evaluations[t, :] = np.exp(current_only_log_denominator)

        for tau in range(t):
            # this loop is only entered if t>0
            mixture_denominator_evaluations[tau, :] += np.exp(current_proposal.logpdf(all_samples[tau, :]))
            mixture_denominator_evaluations[t, :] += np.exp(proposals_over_iterations[tau].logpdf(samples_current))

        log_denominator = - np.log(t + 1) + np.log(mixture_denominator_evaluations[:t + 1, :])

        assert log_numerator.shape == log_denominator.shape

        updated_logweights = log_numerator - log_denominator
        logsumexp_logweights = logsumexp(updated_logweights)
        updated_normalized_logweights = updated_logweights - logsumexp_logweights

        ### Estimate of the normalization constant
        estimate_Z = np.exp(logsumexp_logweights) / (
                    M * (t + 1))  # weights need to have only pi_tilde in the numerator (no escort)
        all_estimate_Z[t] = estimate_Z

        ### metrics
        current_only_logweights = evaluations_target_logpdf[t, :] - current_only_log_denominator
        current_only_normalized_logweights = current_only_logweights - logsumexp(current_only_logweights)

        ### Update proposal

        samples_up_to_now = all_samples[:t + 1, :, :]

        # weights need to have the escort of pi_tilde in the numerator
        updated_escort_logweights = alpha * log_numerator - log_denominator
        updated_normalized_escort_logweights = updated_escort_logweights - logsumexp(updated_escort_logweights)

        # Get alpha-ESS evaluations
        current_only_alphaESS = np.exp(logsumexp(alpha * current_only_normalized_logweights)) ** (1 / (1 - alpha))
        current_only_ESS = 1 / np.exp(logsumexp(2 * current_only_normalized_logweights))

        all_alphaESS[t] = (1 / M) * current_only_alphaESS
        all_ESS[t] = (1 / M) * current_only_ESS
        all_dof[t] = dof_proposal

        # first_moment = np.zeros(D)
        # secnd_moment = np.zeros((D,D))

        # for tau in range(t+1):
        #     for m in range(M):
        #         w = np.exp(updated_normalized_escort_logweights[tau, m])
        #         first_moment += w * samples_up_to_now[tau, m, :]
        #         secnd_moment += w * (samples_up_to_now[tau, m, :].reshape(-1, 1) @ samples_up_to_now[tau, m, :].reshape(1, -1))

        # mu_current = first_moment
        # shape_current =  secnd_moment - (mu_current.reshape(-1, 1) @ mu_current.reshape(1, -1))


        ##########
        # Now optimize dof !

        if t>0:
            
            observed_dof = np.append(observed_dof, np.array([dof_proposal]), axis=0)
            observed_ess = np.append(observed_ess, np.array([np.log(1 -(1 / M) * current_only_alphaESS)]), axis=0)

            if t < num_initial_dof_points :
                dof_proposal = initial_dof_points[t].item()
            else:
                if gpy_model is None:
                    #declare model
                    gpy_model = GPy.models.GPRegression(observed_dof.reshape(-1,1), observed_ess.reshape(-1,1))
                    emukit_model = GPyModelWrapper(gpy_model)
          
                else:
                    emukit_model.set_data(observed_dof.reshape(-1,1), observed_ess.reshape(-1,1))#update

                # gpy_model.kern.lengthscale.fix(0.5) #dummy values for now
                # gpy_model.kern.variance.fix(1e-1)
                # gpy_model.likelihood.variance.fix(1e-1)

                # gpy_model.plot()
                # plt.show()
                
                

                beta_param = 2 * np.log((t**2 + 1)*10 / np.sqrt(2*np.pi)) #from Garnett2023 p 229
                
                acquisition = acquisition_functions['LCB'](emukit_model, beta=beta_param) #high beta => exploration, while small beta => exploitation
                
                optimizer = GradientAcquisitionOptimizer(dof_proposal_space)
                x_new, _ = optimizer.optimize(acquisition)
                
                dof_proposal = x_new.item()
                
                

        

        W = np.exp(updated_normalized_escort_logweights)
        mu_current = np.einsum('tmd,tm->d', samples_up_to_now, W)
        secnd_moment = np.einsum('tm, tmd, tme -> de', W, samples_up_to_now, samples_up_to_now)
        shape_current = secnd_moment - (mu_current.reshape(-1, 1) @ mu_current.reshape(1, -1))


    return all_estimate_Z, all_alphaESS, all_ESS, all_dof


