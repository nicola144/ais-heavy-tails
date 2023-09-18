import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logsumexp

def compute_ess(logw):
  """Compute Effective Sample Size (ESS) using log importance weights."""
  return 1 / np.exp(logsumexp(2 * (logw - logsumexp(logw))))

def save(obj, filename):
    """Save compiled models for reuse."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    """Reload compiled models for reuse."""
    return pickle.load(open(filename, 'rb'))

def safe_gp_optimization(
        bo_model, lower_bound_var=1e-05, upper_bound_var=2.0, bound_len=20.0
    ):
        if bo_model.model.kern.variance[0] < lower_bound_var:
            print("safe optimization: resetting kernel var")
            bo_model.model.kern.variance[0] = 1.0

        if bo_model.model.kern.lengthscale[0] > bound_len:
            print("safe optimization: resetting kernel lengthscale")
            bo_model.model.kern.lengthscale[0] = 1.0

        if bo_model.model.likelihood.variance[0] > upper_bound_var:
            print("safe optimization: resetting lik var")
            bo_model.model.likelihood.variance[0] = upper_bound_var

        if bo_model.model.likelihood.variance[0] < lower_bound_var:
            print("safe optimization: resetting lik var")
            bo_model.model.likelihood.variance[0] = 1.


def plot_contour_lines(dist1, dist2, iteration=0):
    # Generate a grid of points
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Calculate the probability density for each distribution at each point on the grid
    Z1 = dist1.pdf(positions.T)
    Z2 = dist2.pdf(positions.T)

    # Reshape the probability density values to match the grid shape
    Z1 = Z1.reshape(X.shape)
    Z2 = Z2.reshape(X.shape)

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the contour lines for the first distribution in blue color
    ax.contour(X, Y, Z1, colors='blue', label='Proposal')

    # Plot the contour lines for the second distribution in red color
    ax.contour(X, Y, Z2, colors='red', label='Target')

    # Set axis labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Contour Lines')

    # # Add a legend in the upper right corner
    # ax.legend()

    plt.savefig("results/ais-heavy-iter-{}.png".format(iteration), bbox_inches="tight", dpi=100)
    # plt.show()