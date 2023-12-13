# An implementation of stochastic gradient methods for GLMs.


sgd <- function(data, sgd.method, lr, npass=1, lambda=0, ...) {
  # Start tracking time and memory
  start_time <- Sys.time()
  mem_start <- memory.size()

  # Check input.
  stopifnot(
    all(is.element(c("X", "Y", "model"), names(data))),
    sgd.method %in% c("SGD", "ASGD", "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD", "SVRG", "ADAM", "RMSprop", "AMSGrad")
  )
  
  # Initialize constants.
  beta1 <- 0.9
  beta2 <- 0.999
  epsilon <- 1e-8
  m <- v <- vhat <- cache <- rep(0, ncol(data$X))

  # Initialize parameter matrix for the stochastic gradient descent.
  n <- nrow(data$X)
  p <- ncol(data$X)
  niters <- n * npass
  theta.sgd <- matrix(0, nrow = p, ncol = niters + 1)

  # Adjust iterations if method is SVRG
  if (sgd.method == "SVRG") {
    stopifnot(npass %% 2 == 0)
    m <- 2 * n
    niters <- npass / 2 # 2-passes over the data
    colnames(theta.sgd) <- 0:niters * m + 1
  }
    
  lr_sum <- 0
  min_lr <- Inf
  max_lr <- 0
  # Run the stochastic gradient method.
  for (i in 1:niters) {
    idx <- ifelse(i %% n == 0, n, i %% n)
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    theta.old <- theta.sgd[, i]
    ai <- lr(i, ...)
    max_lr <- max(max_lr, ai)
    min_lr <- min(min_lr, ai)
    lr_sum <- lr_sum + ai
    # Update theta based on the method chosen
    theta.new <- switch(sgd.method,
                        "SGD" = sgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "ASGD" = sgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "LS-SGD" = sgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "ISGD" = isgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "AI-SGD" = isgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "LS-ISGD" = isgd.update(theta.old, xi, yi, ai, lambda, data$model),
                        "SVRG" = svrg.update(theta.old, data, lr, lambda, data$model, m, ...),
                        "ADAM" = adam.update(theta.old, xi, yi, ai, lambda, data$model, beta1, beta2, epsilon, m, v, i)$theta,
                        "RMSprop" = rmsprop.update(theta.old, xi, yi, ai, lambda, data$model, beta1, epsilon, cache)$theta,
                        "AMSGrad" = amsgrad.update(theta.old, xi, yi, ai, lambda, data$model, beta1, beta2, epsilon, m, v, vhat, i)$theta
                       )
    theta.sgd[, i + 1] <- theta.new
  }
  lr_avg <- lr_sum / niters

  mem_end <- memory.size()
  memory_usage <- mem_end - mem_start
  # Post-processing for certain methods
  if (sgd.method %in% c("ASGD", "AI-SGD")) {
    theta.sgd <- average.post(theta.sgd)
  }
  if (sgd.method %in% c("LS-SGD", "LS-ISGD")) {
    theta.sgd <- ls.post(theta.sgd, data)
  }

  # End tracking time and memory
  end_time <- Sys.time()
  execution_time <- difftime(end_time, start_time, units = "secs")
  
  cat(" Minimum learning rate:", min_lr, "  Maximum learning rate:", max_lr, "  Average learning rate:", lr_avg, "\n")
  cat("Method:", sgd.method, "  Execution time:", execution_time, "\n")
  return(theta.sgd)
}

# [Include the rest of your update and post-processing functions here...]



################################################################################
# Update functions
################################################################################
sgd.update <- function(theta.old, xi, yi, ai, lambda, glm.model) {
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  theta.new <- theta.old + ai * score(theta.old)
  return(theta.new)
}
isgd.update <- function(theta.old, xi, yi, ai, lambda, glm.model) {
  # Make computation easier.
  xi.norm <- sum(xi^2)
  lpred <- sum(xi * theta.old)
  get.score.coeff <- function(ksi) {
    # Returns:
    #   The scalar value yi - h(θ_i' xi + xi^2 ξ) + λ*||θ_i+ξ||_2
    yi - glm.model$h(lpred + xi.norm * ksi) + lambda*sqrt(sum((theta.old+ksi)^2))
  }
  # 1. Define the search interval.
  ri <- ai * get.score.coeff(0)
  Bi <- c(0, ri)
  if (ri < 0) {
    Bi <- c(ri, 0)
  }
  implicit.fn <- function(u) {
    u - ai * get.score.coeff(u)
  }
  # 2. Solve implicit equation.
  ksi.new <- NA
  if (Bi[2] != Bi[1]) {
    ksi.new <- uniroot(implicit.fn, interval=Bi)$root
  }
  else {
    ksi.new <- Bi[1]
  }
  theta.new <- theta.old + ksi.new * xi
  return(theta.new)
}
svrg.update <- function(theta.old, data, lr, lambda, glm.model, m, ...) {
  n <- nrow(data$X)
  p <- ncol(data$X)
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  # Do one pass of data to obtain the average gradient.
  mu <- rep(0, p)
  for (idx in 1:n) {
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    mu <- mu + score(theta.old)/n
  }
  # Run inner loop, updating w by using a random sample.
  w <- theta.old
  for (mi in 1:m) {
    idx <- sample(1:n, 1)
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    ai <- lr(mi, ...)
    w <- w + ai * (score(w) - score(theta.old) + mu)
  }
  # Assign SGD iterate to the last updated weight ("option I").
  theta.new <- w
  return(theta.new)
}

adam.update <- function(theta.old, xi, yi, ai, lambda, glm.model, beta1, beta2, epsilon, m, v, t) {
  # Compute the gradients (using the score function from your existing code).
  g <- (yi - glm.model$h(sum(xi * theta.old))) * xi - lambda * theta.old

  # Update biased first and second moment estimates
  m <- beta1 * m + (1 - beta1) * g
  v <- beta2 * v + (1 - beta2) * (g^2)

  # Correct bias in first and second moments
  mhat <- m / (1 - beta1^t)
  vhat <- v / (1 - beta2^t)

  # Update the parameters
  theta.new <- theta.old - ai * mhat / (sqrt(vhat) + epsilon)

  return(list(theta = theta.new, m = m, v = v))
}

rmsprop.update <- function(theta.old, xi, yi, ai, lambda, glm.model, decay_rate, epsilon, cache) {
  # Compute the gradients
  g <- (yi - glm.model$h(sum(xi * theta.old))) * xi - lambda * theta.old

  # Update cache with square of gradients
  cache <- decay_rate * cache + (1 - decay_rate) * g^2

  # Update parameters
  theta.new <- theta.old - ai * g / (sqrt(cache) + epsilon)

  return(list(theta = theta.new, cache = cache))
}



amsgrad.update <- function(theta.old, xi, yi, ai, lambda, glm.model, beta1, beta2, epsilon, m, v, vhat, t) {
  # Compute the gradients
  g <- (yi - glm.model$h(sum(xi * theta.old))) * xi - lambda * theta.old

  # Update biased first and second moment estimates
  m <- beta1 * m + (1 - beta1) * g
  v <- beta2 * v + (1 - beta2) * (g^2)

  # Update vhat
  vhat <- pmax(vhat, v)

  # Update parameters
  theta.new <- theta.old - ai * m / (sqrt(vhat) + epsilon)

  return(list(theta = theta.new, m = m, v = v, vhat = vhat))
}

################################################################################
# Post-processing functions
################################################################################
average.post <- function(theta.sgd) {
  return(t(apply(theta.sgd, 1, function(x) {
    cumsum(x)/(1:length(x))
  })))
}
ls.post <- function(theta.sgd, data) {
  # TODO: Generalize beyond Normal(0, A) data.
  n <- nrow(data$X)
  p <- ncol(data$X)
  ncol.theta <- ncol(theta.sgd) # n*npass+1
  # TODO: Generating y can be faster by doing matrix multiplication instead.
  # TODO: Benchmark this compared to forming y within the main SGD loop. The
  # latter method does not have to load in the DATA object into a function,
  # which is expensive.
  # Also the indices are probably off here: y[, 1] should not be all 0 (?).
  y <- matrix(0, nrow=p, ncol=ncol.theta)
  for (i in 1:(ncol.theta-1)) {
    idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
    xi <- data$X[idx, ]
    theta.old <- theta.sgd[, i]
    y[, i+1] <- data$obs.data$A %*% (xi - theta.old)
  }

  beta.0 <- matrix(0, nrow=p, ncol=ncol.theta)
  beta.1 <- matrix(0, nrow=p, ncol=ncol.theta)
  for (i in 1:ncol.theta) {
    x.i <- theta.sgd[, 1:i]
    y.i <- y[, 1:i]
    bar.x.i <- rowMeans(x.i)
    bar.y.i <- rowMeans(y.i)
    beta.1[, i] <- rowSums(y.i*(x.i - bar.x.i))/rowSums((x.i - bar.x.i)^2)
    beta.0[, i] <- bar.y.i - beta.1[, i] * bar.x.i
  }
  return(-beta.0/beta.1)
}
