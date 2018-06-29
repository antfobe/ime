if(!require("install.load")) {install.packages("install.load"); library("install.load")}

pkgs <- c('keras', 'tidyverse')
lapply(pkgs, FUN = install_load);

iris %>% as_tibble %>% gather(feature, value, -Species) %>%
    ggplot(aes(x = feature, y = value, fill = Species)) +
    geom_violin(alpha = 0.5, scale = "width", position = position_dodge(width = 0.9)) +
    geom_boxplot(alpha = 0.5, width = 0.2, position = position_dodge(width = 0.9)) +
    theme_bw()

nn_dat = iris %>% as_tibble %>%
    mutate(sepal_l_feat = scale(Sepal.Length),
           sepal_w_feat = scale(Sepal.Width),
           petal_l_feat = scale(Petal.Length),
           petal_w_feat = scale(Petal.Width),          
           class_num    = as.numeric(Species) - 1, # factor, so = 0, 1, 2
           class_label  = Species) %>%
    select(contains("feat"), class_num, class_label)
nn_dat %>% head(3)

test_f = 0.20
nn_dat = nn_dat %>%
    mutate(partition = sample(c('train','test'), nrow(.), replace = TRUE, prob = c(1 - test_f, test_f)))

x_train = nn_dat %>% filter(partition == 'train') %>% select(contains("feat")) %>% as.matrix
y_train = nn_dat %>% filter(partition == 'train') %>% pull(class_num) %>% to_categorical(3)
x_test  = nn_dat %>% filter(partition == 'test')  %>% select(contains("feat")) %>% as.matrix
y_test  = nn_dat %>% filter(partition == 'test')  %>% pull(class_num) %>% to_categorical(3)

model = keras_model_sequential()
model %>% 
    layer_dense(units = 4, activation = 'relu', input_shape = 4) %>% 
    layer_dense(units = 3, activation = 'softmax')
model %>% summary

model %>% compile(
    loss      = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics   = c('accuracy')
)

history = model %>% fit(
    x = x_train, y = y_train,
    epochs           = 200,
    batch_size       = 20,
    validation_split = 0
)
plot(history)

perf = model %>% evaluate(x_test, y_test)
print(perf)

plot_dat = nn_dat %>% filter(partition == 'test') %>%
    mutate(class_num = factor(class_num),
           y_pred    = factor(predict_classes(model, x_test)),
           Correct   = factor(ifelse(class_num == y_pred, "Yes", "No")))
plot_dat %>% select(-contains("feat")) %>% head(3)

title     = "Classification Performance of Artificial Neural Network"
sub_title = str_c("Accuracy = ", round(perf$acc, 3) * 100, "%")
x_lab     = "True iris class"
y_lab     = "Predicted iris class"
plot_dat %>% ggplot(aes(x = class_num, y = y_pred, colour = Correct)) +
    geom_jitter() +
    scale_x_discrete(labels = levels(nn_dat$class_label)) +
    scale_y_discrete(labels = levels(nn_dat$class_label)) +
    theme_bw() +
    labs(title = title, subtitle = sub_title, x = x_lab, y = y_lab)