#!/usr/bin/R

## Heavily based on https://github.com/leonjessen/keras_tensorflow_on_iris
## Ultimate kudos to you Leon Jessen

## Add packages
if(!require("install.load")) {install.packages("install.load"); library("install.load")}
lapply(c('keras', 'tidyverse'), FUN = install_load);

## Load data, remove "id" field
data <- subset(read.csv(file = "train.csv"), select = c(-id));
test <- subset(read.csv(file = "test.csv"), select = c(-id));

## Prepare/encode data, non-numeric fields as numeric (positive integer), scales, etc
nn_dat = data %>% as_tibble %>% 
    mutate(age_feat = scale(age),
           job_feat = scale(as.numeric(job)),
           marital_feat = scale(as.numeric(marital)),
           education_feat = scale(as.numeric(education)),
           default_feat = scale(as.numeric(default)),
           housing_feat = scale(as.numeric(housing)),
           loan_feat = scale(as.numeric(loan)),
           contact_feat = scale(as.numeric(contact)),
           month_feat = scale(month),
           weekday_feat = scale(day_of_week),
           campaign_feat = scale(campaign),
           pdays_feat = scale(pdays),
           previous_feat = scale(previous),
           poutcome_feat = scale(as.numeric(poutcome)),
           employmnt_feat = scale(emp.var.rate),
           priceidx_feat = scale(cons.price.idx),
           confidx_feat = scale(cons.conf.idx),
           euribor_feat = scale(euribor3m),
           nremployd_feat = scale(nr.employed),
           class_num = y,
           class_label = c("Prediction")) %>%
    select(contains("feat"), class_num, class_label)
nn_dat %>% head(3)

## Partition test dataset & size as 1/5 of all data
test_f = 0.20
nn_dat = nn_dat %>%
    mutate(partition = sample(c('train','test'), nrow(.), replace = TRUE, prob = c(1 - test_f, test_f)))

## Create training and testing data
x_train = nn_dat %>% filter(partition == 'train') %>% select(contains("feat")) %>% as.matrix
y_train = nn_dat %>% filter(partition == 'train') %>% pull(class_num) %>% to_categorical()
x_test  = nn_dat %>% filter(partition == 'test')  %>% select(contains("feat")) %>% as.matrix
y_test  = nn_dat %>% filter(partition == 'test')  %>% pull(class_num) %>% to_categorical()

## Set nn architecture
model = keras_model_sequential()
model %>% 
    layer_dense(units = 19, activation = 'relu', input_shape = 19) %>% 
    layer_dense(units = 2, activation = 'softmax')
model %>% summary

## Compile architecture
model %>% compile(
    loss      = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics   = c('accuracy')
)

## And then we train :)
history = model %>% fit(
    x = x_train, y = y_train,
    epochs           = 200,
    batch_size       = 20,
    validation_split = 0
)
plot(history)

## Output performance
perf = model %>% evaluate(x_test, y_test)
print(perf)

## Prepare nn_dat for plotting
plot_dat = nn_dat %>% filter(partition == 'test') %>%
    mutate(class_num = factor(class_num),
           y_pred    = factor(predict_classes(model, x_test)),
           Correct   = factor(ifelse(class_num == y_pred, "Yes", "No")))
plot_dat %>% select(-contains("feat")) %>% head(3)

## Plot confusion matrix
title     = "Classification Performance of Artificial Neural Network"
sub_title = str_c("Accuracy = ", round(perf$acc, 3) * 100, "%")
x_lab     = "True Outcome"
y_lab     = "Predicted Outcome"
plot_dat %>% ggplot(aes(x = class_num, y = y_pred, colour = Correct)) +
    geom_jitter() +
    scale_x_discrete(labels = levels(nn_dat$class_label)) +
    scale_y_discrete(labels = levels(nn_dat$class_label)) +
    theme_bw() +
    labs(title = title, subtitle = sub_title, x = x_lab, y = y_lab)

## Fit 'official' test data
nn_test = test %>% as_tibble %>% 
    mutate(age_feat = scale(age),
           job_feat = scale(as.numeric(job)),
           marital_feat = scale(as.numeric(marital)),
           education_feat = scale(as.numeric(education)),
           default_feat = scale(as.numeric(default)),
           housing_feat = scale(as.numeric(housing)),
           loan_feat = scale(as.numeric(loan)),
           contact_feat = scale(as.numeric(contact)),
           month_feat = scale(month),
           weekday_feat = scale(day_of_week),
           campaign_feat = scale(campaign),
           pdays_feat = scale(pdays),
           previous_feat = scale(previous),
           poutcome_feat = scale(as.numeric(poutcome)),
           employmnt_feat = scale(emp.var.rate),
           priceidx_feat = scale(cons.price.idx),
           confidx_feat = scale(cons.conf.idx),
           euribor_feat = scale(euribor3m),
           nremployd_feat = scale(nr.employed)) %>%
    select(contains("feat"))

t_test  = nn_test %>% select(contains("feat")) %>% as.matrix
pred = model %>% predict(t_test)

## Ouput to .csv submission file
write.csv(data.frame(id = 1:nrow(pred) - 1, y = round(pred[,2])), file = "r-nn-submission.csv", row.names = FALSE)