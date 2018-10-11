if(!require("install.load")) {install.packages("install.load"); library("install.load")}
lapply(c('TSP', 'tspmeta', 'igraph'), 
       FUN = install_load);

coords.df <- data.frame(long=runif(10, min=0, max=2), lat=runif(10, min=0, max=2))
coords.mx <- as.matrix(coords.df)

# Compute distance matrix
dist.mx <- dist(coords.mx)

# Calculate (one iteration) of minimal edges
minedge <- function(dstmx) {
  matrix(
    unlist(
      lapply(1:10, FUN = function(X){ 
        return(
          c(X, 
            which(
              dstmx[X,] == min(dstmx[X, ][dstmx[X, ] != 0]), 
              arr.ind = TRUE)[1,2] - 1
          )
        )
      })
    ), 
    ncol = 2,
    byrow = TRUE
  )
}

# Construct a TSP object
tsp.ins <- tsp_instance(coords.mx, dist.mx)
tour <- run_solver(tsp.ins, method="2-opt")

#Plot
autoplot(tsp.ins, tour)

# The idea here is to make & update a distance array calculated 
# by how much a node contributes to the overall path distance
# I think that if for every edge added this distance array is recalculated
# the best (shortest) path may be found iteratively
# In other words - if an edge is made it's distance is discounted from the
# 'missing' path, what we want is the path with less distance
mx <- data.frame(matrix(nrow = nrow(coords.mx), ncol = nrow(coords.mx)+1)); 
colnames(mx) <- c("sumd", 1:nrow(coords.mx))
for (i in 1:nrow(mx)) {
  mx[i,1] <- 0
  for (j in 1:nrow(mx)) {
    mx[i, j + 1] <- sqrt(sum((coords.mx[i, ] - coords.mx[j, ])^2))
    mx[i,1] <- mx [i,1] + mx[i, j + 1]
  }
}