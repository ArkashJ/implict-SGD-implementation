# Form plots displaying error of AI-SGD using various learning rates, one of
# which uses the parameters obtained from tuning.

library(dplyr)
library(ggplot2)
library(mvtnorm)

source("functions.R")
source("sgd.R")
source("theory/optimal_aisgd.R")

# Set this to TRUE in order to have the following code also tune the parameters
bool.tune <- FALSE

# Construct functions for learning rate.
lr <- function(n, par) {
  # Ruppert's learning rate.
  # Note:
  # α / (α + n) = 1 / (1 + lambda0*n), where lambda0 = 1/α
  D <- par[1]
  alpha <- par[2]
  D*n^alpha
}



methods_to_test <- c("ADAM", "AMSGrad", "RMSprop", "ASGD", "ISGD")

################################################################################
# Normal, n=1e5, p=1e2
################################################################################
if (bool.tune) {
  set.seed(42)
  n <- 1e5
  p <- 1e2
  model <- "gaussian"
  X.list <- generate.X.A(n, p)
  d <- generate.data(X.list, theta=rep(5, p), glm.model=get.glm.model(model))
  vals <- tunePar(d, lr=lr)
}

pars <- rbind(c(0.001,0), c(0.005, 0))
run("gaussian", pars=pars, n=1e4, p=1e2, add.methods=methods_to_test)

################################################################################
# Poisson, n=1e4, p=10
################################################################################
if (bool.tune) {
  set.seed(42)
  n <- 1e4
  p <- 1e1
  model <- "poisson"
  X.list <- generate.X.A(n, p)
  d <- generate.data(X.list, glm.model=get.glm.model(model), theta=2 * exp(-seq(1, p)))
  vals <- tunePar(d, lr=lr)
}

pars <- rbind(c(10, -0.8), c(1/0.01, -1))
run("poisson", pars=pars, n=1e4, p=1e1, add.methods=methods_to_test)

################################################################################
# Logistic, n=1e4, p=1e2
################################################################################
if (bool.tune) {
  set.seed(42)
  n <- 1e4
  p <- 1e2
  model <- "logistic"
  X.list <- generate.X.A(n, p)
  d <- generate.data(X.list, glm.model=get.glm.model(model), theta=2 * exp(-seq(1, p)))
  vals <- tunePar(d, lr=lr)
}

pars <- rbind(c(0.892, -0.5), c(1/0.01, -1))
run("logistic", pars=pars, n=1e4, p=1e2, add.methods=methods_to_test)
