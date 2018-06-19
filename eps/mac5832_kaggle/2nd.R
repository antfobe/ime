if(!require("install.load")) {install.packages("install.load"); library("install.load")}

pkgs <- c('foreach', 'doParallel', 'parallel', 'e1071', 'parallelSVM')
lapply(pkgs, FUN = function(X){
                        if(!X %in% installed.packages()) {
                            install_load(X)
                        }
                    });

doParallel::registerDoParallel(cores = parallel::detectCores() - 1);

data <- read.csv(file = "train.csv");
test <- read.csv(file = "test.csv");
## encode data to apply learning methods

conv2numeric <- c("job", "marital", "education", "default", "housing", "loan", "contact", "poutcome");
data[, conv2numeric] <- sapply(data[, conv2numeric], FUN = as.numeric);
test[, conv2numeric] <- sapply(test[, conv2numeric], FUN = as.numeric);
# data$education <- as.numeric(data$education);
# data$job <- as.numeric(data$job);
# data$marital <- as.numeric(data$marital);
# data$default <- as.numeric(data$default);
# data$housing <- as.numeric(data$housing);
# data$loan <- as.numeric(data$loan);
# data$contact <- as.numeric(data$contact);
# data$poutcome <- as.numeric(data$poutcome);

x <- subset(data, select = c(-y,-poutcome));
y <- subset(data, select = y);
t <- subset(test, select = c(-poutcome));
##y <- as.logical(y$y);

svm_model <- e1071::svm(y$y ~ ., data = x);
summary(svm_model);

system.time(pred <- predict(svm_model, t));

write.csv(data.frame(id = t$id, pred = pred), file = "submission-nontuned.csv", row.names = FALSE);
write.csv(data.frame(id = t$id, pred = round(pred)), file = "submission-nontuned-rounded.csv", row.names = FALSE);

svm_tune <- e1071::tune(svm, train.x = x, train.y = y, 
                 kernel = "radial", ranges = list(cost = 10^(-1:2), gamma = c(.5,1,2)));

svm_model_after_tune <- e1071::svm(y ~ ., data = x, kernel = "radial",
                            cost = svm_tune$best.parameters$cost, gamma = svm_tune$best.parameters$gamma);

system.time(pred <-predict(svm_model_after_tune, t));

table(round(pred), y$y);

write.csv(data.frame(id = t$id, pred = pred), file = "submission-tuned.csv", row.names = FALSE);
write.csv(data.frame(id = t$id, pred = round(pred)), file = "submission-tuned-rounded.csv", row.names = FALSE);
