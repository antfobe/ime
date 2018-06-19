if(!require("install.load")) {install.packages("install.load"); library("install.load");}

pkgs <- c('neuralnet');
lapply(pkgs, FUN = function(X){
    if(!X %in% installed.packages()) {
        install_load(X);
    }
});

data <- read.csv(file = "train.csv");
test <- read.csv(file = "test.csv");

## encode data to apply learning methods
conv2numeric <- c("job", "marital", "education", "default", "housing", "loan", "contact", "poutcome");
data[, conv2numeric] <- sapply(data[, conv2numeric], FUN = as.numeric);
test[, conv2numeric] <- sapply(test[, conv2numeric], FUN = as.numeric);

# Random sampling
samplesize = 0.60 * nrow(data);
set.seed(80);
index = sample(
    seq_len(
        nrow(data)), size = samplesize);

# Create training and test set
datatrain = data[index,];
datatest = data[-index,];

## scale to fit nn, as described in https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
max = apply(data, 2, max);
min = apply(data, 2, min);
scaled = as.data.frame(scale(data, center = min, scale = max - min));

# creating training and test set
trainNN = scaled[index,];
testNN = scaled[-index,];

# fit neural network
set.seed(2);
n <- names(trainNN);
f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")));
NN = neuralnet(formula = f, data = trainNN, hidden = c(12, 8, 5, 3), linear.output = T);

# plot neural network
plot(NN);