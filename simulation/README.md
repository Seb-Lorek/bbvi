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
- We aim to assess the comparative accuracy of BBVI against MCMC in approximating the posterior distribution over a predefined time window. This analysis will focus on location-scale regression, incorporating additive linear predictors.
- Compare the approximation of smooth functions and the related (gloabl) inverse smoothing parameters from BBVI and MCMC.

## Data generating mechanism 

- Generate random data set with different relationships in the linear predictors of interest.
- Start by generating random data for the regressor from a uniform distribution, parametrized by its minimum and maximum value. 
- After this we will generate the data for the linear predictors by applying functions on the random covariate vector, to model the additive structure in the linear predictor.
- Finally, conditional on the parameters of the response distribution we sample the response after applying the inverse link function on the additive linear predictor.

## Estimand/target of analysis 

- Start targeting estimands for fixed linear predictor specifications.
- Then we target the posterior distributions of MCMC and BBVI.
- Finally we target smooth functions and the corresponding direction of the (inverse) smoothing parameter. 

## Methods 

- Compare estimands to true fixed coefficients.
- Use kernel density plots as a visual tool and the Kullback-Leibler divergence as a formal measure to distinguish closeness to the "true" posterior.
- Compare estimated smooth functions to the true underlying function and assess the direction of the (inverse) smoothing parameter. 

## Performance measures

- Bias, EmpSE (also boxplots) and their related Monte Carlo SE.
- Kullback-Leibler divergence.
- MSE and its related Monte Carlo SE.

## Dev Note 

Run the python scripts from the project directory `bbvi`.

```
python3 simulation/main_sim.py
```