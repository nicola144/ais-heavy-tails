data {
    int<lower=0> N;     // number of data points
    int<lower=0> K;     // number of covariates
    matrix[N, K] X;     // matrix of covariates
    vector[N] y;        // response variable
}
parameters {
    real alpha;         // intercept
    vector[K] beta;     // coefficients for covariates
}
model {
    // Priors
    alpha ~ cauchy(0, 1);
    for (k in 1:K) {
        beta[k] ~ cauchy(0, 1);
    }

    // Likelihood with student-t errors, 5 degrees of freedom
    y ~ student_t(3, alpha + X * beta, 1);
}
