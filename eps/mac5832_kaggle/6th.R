# !/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# test if there is only one argument: if not, return an error
if (length(args)!=3) {
  stop("Usage: Rscript --vanilla 6th.R <datatrain.csv> <datatest.csv> <y-row_name>.n", call.=FALSE)
}

if(!require("install.load")) {install.packages("install.load"); library("install.load")}

lapply(c('foreach', 'doParallel', 'parallel', 'e1071', 'parallelSVM', 'caret', 'pROC'), 
       FUN = install_load);

cat("Setting\tseed ...\n\n");
svmseed <- as.numeric(format(Sys.Date(), "%Y"));
set.seed(svmseed);

doParallel::registerDoParallel(cores = parallel::detectCores()/2);

cat("Loading\tdata ...\n\n");
data <- read.csv(file = args[1]);
test <- read.csv(file = args[2]);

## encode data to apply learning methods
# conv2numeric <- c("job", "marital", "education", "default", "housing", "loan", "contact", "poutcome");
# data[, conv2numeric] <- sapply(data[, conv2numeric], FUN = as.numeric);
# test[, conv2numeric] <- sapply(test[, conv2numeric], FUN = as.numeric);

x <- subset(data, select = -which(names(data) %in% c(args[3], names(data)[1])));
y <- subset(data, select = args[3]);
## woah - cannot use just 't', apparently messes with built-in functions: 
## https://stats.stackexchange.com/questions/233531/object-of-type-closure-is-not-subsettable
# test_t <- subset(test, select = c(-id));
test_t <- subset(test, select = -which(names(test) %in% c(args[3], names(test)[1])));
# xsim <- x[sample(nrow(x), nrow(y)),];

cat("Training model ...\n\n");
system.time(svm_model <- parallelSVM::parallelSVM(x, y[,1],
                                      type = "C-classification",
                                      kernel = "radial",
                                      seed = svmseed,
                                      probability = TRUE, 
                                      gamma = 0.004487103, cost = 3,
                                      numberCores = parallel::detectCores()-1));

## performance
system.time(pred <- predict(svm_model, x, decision.values = TRUE));

cat("\nPrediction summary: \n");
cat(c(names(summary(pred)), "\n", summary(pred), "\n"))
pred <- attributes(pred);

##data.frame(matrix(pred$probabilities[y$y != round(pred_numeric)], ncol = 2), 
##y$y[y$y != round(pred_numeric)])

system.time(pred <- predict(svm_model, test_t, decision.values = TRUE));
pred <- attributes(pred);
# pred_numeric <- sapply(1:nrow(pred$probabilities), 
#                        FUN = function(X) {
#                          pred$probabilities[X,][names(pred$probabilities[X,]) == 1]});

write.csv(data.frame(id = 1:nrow(test), pred = as.character(pred)), file = "parallel-nontuned.csv", row.names = FALSE);

## Tunning
### SPLIT DATA INTO K FOLDS ###
## sets seed as 4 digits current year

x$fold <- caret::createFolds(1:nrow(x), k = 10, list = FALSE);
### PARAMETER LIST ###
parms <- expand.grid(cost = c(seq(from = 0.0003, to = 0.0005, by = 0.00001)), gamma = c(seq(from = 1.05, to = 1.20, by = 0.005)));
### LOOP THROUGH PARAMETER VALUES ###
result <- foreach::foreach(i = 1:nrow(parms), .combine = rbind) %do% {
    c <- parms[i, ]$cost;
    g <- parms[i, ]$gamma;
    ### K-FOLD VALIDATION ###
    out <- foreach::foreach(j = 1:max(x$fold), .combine = rbind, .inorder = FALSE) %dopar% {
        deve <- x[x$fold != j, ];
        test <- x[x$fold == j, ];
        mdl <- e1071::svm(y[1:nrow(deve),] ~ ., data = deve, type = "C-classification", 
                          kernel = "radial", cost = c, gamma = g, probability = TRUE);
        pred <- predict(mdl, test, decision.values = TRUE, probability = TRUE);
        data.frame(y = y[1:nrow(test),], prob = attributes(pred)$probabilities[, 2]);
    }
    ### CALCULATE SVM PERFORMANCE ###
    roc <- pROC::roc(as.factor(out$y), out$prob);
    data.frame(parms[i, ], roc = roc$auc[1]);
}

saveRDS(result, "seq-vector_gridsearch-result.rds");

#svm_model_after_tune <- e1071::svm(y$y ~ ., data = x, kernel = "radial",
#                                   cost = svm_tune$best.parameters$cost, gamma = svm_tune$best.parameters$gamma);

#system.time(pred <-predict(svm_model_after_tune, x));
# pred_numeric <- as.double(as.character(pred));
# pred_numeric[pred_numeric < 0] <- 0.0;
# table(round(pred), y$y);

#write.csv(data.frame(id = t$id, pred = pred), file = "submission-tuned.csv", row.names = FALSE);
#write.csv(data.frame(id = t$id, pred = round(pred)), file = "submission-tuned-rounded.csv", row.names = FALSE);
