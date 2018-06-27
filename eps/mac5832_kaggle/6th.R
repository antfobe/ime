if(!require("install.load")) {install.packages("install.load"); library("install.load")}

lapply(c('foreach', 'doParallel', 'parallel', 'e1071', 'parallelSVM', 'caret', 'pROC'), 
       FUN = install_load);

svmseed <- as.numeric(format(Sys.Date(), "%Y"));
set.seed(svmseed);

doParallel::registerDoParallel(cores = parallel::detectCores()-1);

data <- read.csv(file = "train.csv");
test <- read.csv(file = "test.csv");

## encode data to apply learning methods
conv2numeric <- c("job", "marital", "education", "default", "housing", "loan", "contact", "poutcome");
data[, conv2numeric] <- sapply(data[, conv2numeric], FUN = as.numeric);
test[, conv2numeric] <- sapply(test[, conv2numeric], FUN = as.numeric);

x <- subset(data, select = c(-y,-id));
y <- subset(data, select = y);
xsim <- x[sample(nrow(x), nrow(y)/2),];
## woah - cannot use just 't', apparently messes with built-in functions: 
## https://stats.stackexchange.com/questions/233531/object-of-type-closure-is-not-subsettable
test_t <- subset(test, select = c(-id));

system.time(svm_model <- parallelSVM::parallelSVM(xsim, y$y,
                                      type = "C-classification",
                                      kernel = "radial",
                                      seed = svmseed,
                                      probability = TRUE, 
                                      gamma = 0.004487103, cost = 3,
                                      numberCores = parallel::detectCores()-1));

## performance
system.time(pred <- predict(svm_model, x));
pred_numeric <- as.double(as.character(pred));
pred_numeric[pred_numeric < 0] <- 0.0;
cat("Performance : [", 
    length(pred[pred == y$y])/nrow(y), 
    "], #y = ", length(round(pred_numeric)[round(pred_numeric) == 1]),"\n");

system.time(pred <- predict(svm_model, test_t));
pred_numeric <- as.double(as.character(pred));
pred_numeric[pred_numeric < 0] <- 0.0;
cat("#y = ", length(round(pred_numeric)[round(pred_numeric) == 1]),"\n");

write.csv(data.frame(id = test$id, pred = pred_numeric), file = "parallel-nontuned.csv", row.names = FALSE);
write.csv(data.frame(id = test$id, pred = round(pred_numeric)), file = "parallel-nontuned-rounded.csv", row.names = FALSE);

## Tunning
### SPLIT DATA INTO K FOLDS ###
## sets seed as 4 digits current year

x$fold <- caret::createFolds(1:nrow(x), k = 10, list = FALSE);
### PARAMETER LIST ###
parms <- expand.grid(cost = c(2.4, 2.5), gamma = c(0.004518313, 0.0044));
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


#svm_model_after_tune <- e1071::svm(y$y ~ ., data = x, kernel = "radial",
#                                   cost = svm_tune$best.parameters$cost, gamma = svm_tune$best.parameters$gamma);

#system.time(pred <-predict(svm_model_after_tune, x));
# pred_numeric <- as.double(as.character(pred));
# pred_numeric[pred_numeric < 0] <- 0.0;
# table(round(pred), y$y);

#write.csv(data.frame(id = t$id, pred = pred), file = "submission-tuned.csv", row.names = FALSE);
#write.csv(data.frame(id = t$id, pred = round(pred)), file = "submission-tuned-rounded.csv", row.names = FALSE);
