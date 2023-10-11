data {
    int<lower=0> N;  // number of data points
    vector[N] x1;    // first covariate
    vector[N] x2;    // second covariate
    vector[N] x3;    // third covariate
    vector[N] y;     // response variable
}
parameters {
    real alpha;       // intercept
    real beta1;       // coefficient for x1
    real beta2;       // coefficient for x2
    real beta3;       // coefficient for x3
}
model {
    // Priors
    alpha ~ cauchy(0, 1);
    beta1 ~ cauchy(0, 1);
    beta2 ~ cauchy(0, 1);
    beta3 ~ cauchy(0, 1);

    // Likelihood with student-t errors, 5 degrees of freedom
    y ~ student_t(5, alpha + beta1 * x1 + beta2 * x2 + beta3 * x3, 1);
}