# Simulation study for Bayesian semiparametric distributional regression

## General structure 

Follow the ADEMP-structure for simulation studies proposed by (Morris et. al. 2019).
ADEMP is an acronym for:

- Aims
- Data-generating mechanism
- Estimands/target of analysis
- Methods
- Performance measures 

## Aims

- Illustrate the effectiveness of the BBVI algorithm by establishing that the parameters converge in a mean-square fashion to their true values within a frequentist context, for Bayesian linear regression and Bayesian logistic regression.
- Compare the posterior distributions of BBVI and MCMC via density plots.
- Assess the comparative accuracy of BBVI and MCMC in approximating the posterior distribution over a predefined time window/iterations. This analysis will focus on location-scale regression, incorporating additive linear predictors.

## Data generating mechanism 

- Generate random data set with different relationships in the linear predictors of interest.
- Start by generating random data for the regressors from a uniform distribution, parametrized by their minimum and maximum value. 
- After this we will generate the data for the linear predictors by applying functions on the random covariate vectors, to model the additive structure in the linear predictor.
- Finally, conditional on the parameters of the response distribution we sample the response after applying the inverse link function on the additive linear predictor.
- Respect the sampling machanism for different distribution, f.e. normal, bernoulli etc..

## Estimand/target of analysis 

- Start targeting estimands in the linear predictor specifications.
- Then we target the posterior distributions of MCMC and BBVI (other targets).

## Methods 

- Compare estimators of BBVI to true fixed coefficients.
- Use kernel density plots as a visual tool to compare the posteriors of BBVI and MCMC. Furthermore use the Kullback-Leibler divergence as a formal measure to distinguish closeness to the "true" posterior.

## Performance measures

- Bias, EmpSE and their related Monte Carlo SE.
- Kullback-Leibler divergence.

## Dev Note 

Run the python scripts from the project directory `bbvi`.

```
python3 simulation/main_sim.py
```