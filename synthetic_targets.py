import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


def matrix_condition(d, cond):
    # returns a symmetric positive definite matrix of size d with condition number cond
    y = np.random.uniform(-1, 1, d)
    Y = np.eye(d) - (2 / sum(y ** 2)) * np.outer(y, y)
    D = np.diag([np.exp((i / (d - 1)) * np.log(cond)) for i in range(d)])
    return Y @ D @ Y


### defining standard Student distributions

def unnormalized_logpdf_Student(x, dof, loc, inv_shape):
    d = len(x)
    assert len(loc) == d and np.shape(inv_shape) == (d, d)

    gap = x - loc
    return -0.5 * (dof + len(x)) * np.log(1 + (1 / dof) * np.inner(gap, inv_shape @ gap))


def normalization_Student(d, dof, shape):
    return (gamma(dof / 2) * ((dof * np.pi) ** (d / 2)) * np.sqrt(np.linalg.det(shape))) / gamma((dof + d) / 2)


### defining Student versions of the banana-shaped distribution
### dimension must be at least 2

def banana_transform(x, b, c):
    res = np.copy(x)
    res[1] = x[1] + b * (x[0] ** 2 - c ** 2)
    return res


def unnormalized_logpdf_banana(x, dof, b, c):
    d = len(x)
    assert d >= 2
    inv_diag = np.concatenate((np.array([1 / c ** 2]), np.ones(d - 1)))
    return unnormalized_logpdf_Student(banana_transform(x, b, c), dof, np.zeros(d), np.diag(inv_diag))


def normalization_banana(d, dof, b, c):
    assert d >= 2
    diag = np.concatenate((np.array([c ** 2]), np.ones(d - 1)))
    return normalization_Student(d, dof, np.diag(diag))
    # we take advantage of the fact that the "banana transformation" preserves the normalization constant


###defining mixtures of two Student distributions with same shape matrix

def unnormalized_logpdf_mixture(x, dof, loc_1, loc_2, inv_shape):
    d = len(x)
    assert len(loc_1) == d and len(loc_2) == d and np.shape(inv_shape) == (d, d)

    gap_1 = x - loc_1
    gap_2 = x - loc_2

    unnormalized_pdf = (1 + (1 / dof) * np.inner(gap_1, inv_shape @ gap_1)) ** (-0.5 * (dof + d)) + (
                1 + (1 / dof) * np.inner(gap_2, inv_shape @ gap_2)) ** (-0.5 * (dof + d))

    return np.log(0.5 * unnormalized_pdf)


def normalization_mixture(d, dof, shape):
    return normalization_Student(d, dof, shape)


### a plotting function

def plot_unnormalized_density(unnormalized_logpdf):
    nb_points = 500
    X = np.linspace(-20, 20, nb_points)
    Y = np.linspace(-20, 20, nb_points)

    pdf = np.zeros((nb_points, nb_points))
    Z = 0
    for i in range(nb_points):
        for j in range(nb_points):
            log_pi_tilde = np.exp(unnormalized_logpdf([X[i], Y[j]]))
            pdf[j, i] = log_pi_tilde
            Z += log_pi_tilde

    pdf = (1 / Z) * pdf  # normalization

    # plot
    fig, ax = plt.subplots()
    ax.contour(X, Y, pdf)
    plt.show()