
#!/usr/bin/env Rscript
# Compare optimization methods for logistic regression to classify handwritten
# digits from MNIST data. Using Kaggle's split of training and test datasets.
# Kaggle: https://www.kaggle.com/c/digit-recognizer
#
# Dimensions:
#   n=42,000 observations
#   p=784 parameters

library(ggplot2)

source("functions.R")
source("functions_logistic.R")
source("sgd.R")

set.seed(42)
raw <- read.csv("examples/data/train.csv")
idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
raw.train <- raw[idxs,]
raw.test <- raw[-idxs, ]

# Preprocess and normalize data
process_images <- function(raw_data) {
  labels <- raw_data$label
  images <- as.matrix(raw_data[, -1])
  images <- apply(images, 1, function(x) x / 255)
  list(X = images, Y = labels)
}
data.train <- process_images(raw.train)
data.test <- process_images(raw.test)
data.train$model <- get.glm.model("logistic")
data.test$model <- get.glm.model("logistic")

# Build models using training data and output error on test data.
sgd.methods <- c("ADAM", "RMSprop", "AMSGrad")
pars <- c(0.025, 0.025, 0.005, 0.0025, 0.001)
out <- run.logistic(data.train, data.test, sgd.methods, pars, npass=10)
print(out)

