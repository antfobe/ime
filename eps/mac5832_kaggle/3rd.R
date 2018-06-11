if(!require("install.load")) {install.packages("install.load"); library("install.load")}

pkgs <- c('neuralnet')
lapply(pkgs, FUN = function(X){
    if(!X %in% installed.packages()) {
        install_load(X)
    }
});

