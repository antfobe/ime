minedge <- function(dstmx) {
  rows <- nrow(dstmx)
  sums <- sapply(dstmx[1:rows, ], sum)
  edgemx <- matrix(ncol = rows, nrow = rows, data = rep(x = 0, rows * rows))
  for (k in order(sums, decreasing = TRUE)){
    dstmx[k, k] <- Inf
    for (n in order(dstmx[, k])) {
      if (!(n %in% edgemx[, 2])) {
        edgemx[k, 1] = k
        edgemx[k, 2] = n
        break()
      }
    }
  }
  C <- igraph::clusters(igraph::graph_from_edgelist(edgemx))
  
  for (k in 1:C$no) {
    for (m in 1:C$no[1:C$no != k]) {
      1:rows[C$membership == k]
    }
  }
  for(i in 3:rows){
    for(k in order(sapply(1:rows, FUN = function(X) {
      sum(sums[c(edgemx[X, ])])
    }), decreasing = TRUE)) {
      print(k)
      for (n in order(dstmx[, k])) {
        if (!(n %in% edgemx[k, ])) {
          edgemx[k, i] = n
          break()
        }
      }
    }
  }
  dstmx[dstmx == Inf] <- 0.0
  return(edgemx)
}

# sapply(1:nrow(mine), FUN = function(X){
#        sum(diag(as.matrix(mx[c(mine[X, ]), c(mine[X, -1:0], mine[X, 1])])))
# })

extnl <- cx[which(cx == min(cx[1, ]), arr.ind = TRUE)[, 1], ]
list_lextnl <- matrix(cx[c(which(cx[ ,2] > extnl[2], arr.ind = TRUE)), ], ncol = 2)
list_rextnl <- matrix(cx[c(which(cx[ ,2] <= extnl[2], arr.ind = TRUE)), ], ncol = 2)
extnr <- cx[which(cx == max(cx[1, ]), arr.ind = TRUE)[, 1], ]
extnu <- cx[which(cx == max(cx[2, ]), arr.ind = TRUE)[, 1], ]
extnd <- cx[which(cx == min(cx[2, ]), arr.ind = TRUE)[, 1], ]


cx <- t(combn(10,3))
area <- 1:nrow(cx)
for (i in 1:nrow(cx)) {
  area[i] <- abs(coords.mx[cx[i, 1], 1] * (coords.mx[cx[i, 2], 2] - coords.mx[cx[i, 3], 2]) +
                   coords.mx[cx[i, 2], 1] * (coords.mx[cx[i, 3], 2] - coords.mx[cx[i, 1], 2]) +
                   coords.mx[cx[i, 3], 1] * (coords.mx[cx[i, 1], 2] - coords.mx[cx[i, 2], 2]))/2
}
point_dst_to_segment <- function(As, Bs, Cp) {
  mab <- (As[2] - Bs[2]) / (As[1] - Bs[1])
  hxp <- ((As[2] + As[1] * mab - Cp[2]) * mab - Cp[1]) / (mab^2 + 1)
  if(hxp < min(As[1], Bs[1]) || hxp > max(As[1], Bs[1])) {
    return(
      min(sqrt(sum((As - Cp)^2)), 
          sqrt(sum((Bs - Cp)^2)))
    )
  }
  return (as.numeric(
       abs(As[1] * (Bs[2] - Cp[2]) +
           Bs[1] * (Cp[2] - As[2]) +
           Cp[1] * (As[2] - Bs[2])) /
           sqrt(sum((As - Bs)^2))
          ))
}
hull_path <- function(hull_pts, dst_mx) {
  return(
    sum(
      unlist(
        sapply(1:(length(hull_pts)),
               FUN = function(X){
                 dst_mx[hull_pts[X],
                        hull_pts[X+1]]
                 }),
        dst_mx[hull_pts[1],
               hull_pts[length(hull_pts)]]
      )
    )
  )
}
hull_plot <- function(hull){
  plot(coords.mx, cex = 0.5)
  text(coords.mx, labels = as.character(1:nrow(coords.mx)), pos = 1)
  for (i in 1:length(e_hull)) {
    lines(coords.mx[c(e_hull[[i]], e_hull[[i]][1]), ])  
  }
}
hull_init <- function(coords_mx){
  o_hull <- list(chull(coords_mx));
  g_pts <- (1:nrow(coords_mx))[!(1:nrow(coords_mx) %in% o_hull[[1]])];
  while(!is.na(g_pts[1])) {
    lchull <- g_pts[chull(coords_mx[g_pts, ])];
    lchull <- lchull[!is.na(lchull)];
    o_hull <- c(o_hull, list(lchull));
    g_pts <- g_pts[!(g_pts %in% o_hull[[length(o_hull)]])];
  }
  return(o_hull)
}

distance_mx <- function(coords_mx){
  lmx <- data.frame(matrix(nrow = nrow(coords_mx), ncol = nrow(coords_mx))); 
  colnames(lmx) <- 1:nrow(coords_mx);
  for (i in 1:nrow(lmx)) {
    for (j in 1:nrow(lmx)) {
      lmx[i, j] <- sqrt(sum((coords_mx[i, ] - coords_mx[j, ])^2));
    }
  }
  return(lmx)
}

coords.df <- data.frame(long=runif(10, min=0, max=2), lat=runif(10, min=0, max=2));
coords.mx <- as.matrix(coords.df);

e_hull <- hull_init(coords_mx = coords.mx)
mx <- distance_mx(coords.mx)

while (length(e_hull) > 1) {
  while (length(e_hull[[2]]) > 0) {
    print(paste(e_hull))
    lsmx <- matrix(nrow = length(e_hull[[1]]), ncol = length(e_hull[[2]]))
    for (i in 2:nrow(lsmx)) {
      for (j in 1:ncol(lsmx)) {
        lsmx[i, j] <- point_dst_to_segment(as.numeric(coords.mx[e_hull[[1]][i], ]), 
                                           as.numeric(coords.mx[e_hull[[1]][i - 1], ]),
                                           as.numeric(coords.mx[e_hull[[2]][j], ]));
      }
    }
    for (j in 1:ncol(lsmx)) {
      lsmx[1, j] <- point_dst_to_segment(as.numeric(coords.mx[e_hull[[1]][1], ]), 
                                                  as.numeric(coords.mx[e_hull[[1]][length(e_hull[[1]])], ]),
                                                  as.numeric(coords.mx[e_hull[[2]][j], ]));
    }
    
    lmin <- which(lsmx == min(lsmx), arr.ind = TRUE)
    #lmin_subs <- max((lmin[1, 1] + (length(e_hull[[1]]) - 1))  %% length(e_hull[[1]]), 1)
    if (nrow(lmin) > 1) {
      if (sum(c(1, length(e_hull[[1]])) %in% lmin[, 1]) == 2) {
        e_hull[[1]] <- c(e_hull[[2]][as.numeric(lmin[1, 2])], e_hull[[1]])
      } else {
        e_hull[[1]] <- c(e_hull[[1]][1:lmin[1, 1]],
                         e_hull[[2]][as.numeric(lmin[1, 2])],
                         e_hull[[1]][lmin[2, 1]:length(e_hull[[1]])])
      }
      lmin <- lmin[1, ]
    } else {
      lmin_subs <- (lmin[1] - 1 + length(e_hull[[1]])) %% length(e_hull[[1]])
      lmin_plus <- max((lmin[1] + 1) %% length(e_hull[[1]]), 1)
      if(hull_path(c(e_hull[[1]][1:lmin_subs],
                     e_hull[[2]][as.numeric(lmin[2])],
                     e_hull[[1]][lmin[1]:length(e_hull[[1]])]), mx)
        > hull_path(c(e_hull[[1]][1:lmin[1]],
                     e_hull[[2]][as.numeric(lmin[2])],
                     e_hull[[1]][lmin_plus:length(e_hull[[1]])]), mx))
      {
        e_hull[[1]] <- c(e_hull[[1]][1:lmin_subs],
                       e_hull[[2]][as.numeric(lmin[2])],
                       e_hull[[1]][lmin[1]:length(e_hull[[1]])])
      } else {
        e_hull[[1]] <- c(e_hull[[1]][1:lmin[1]],
                         e_hull[[2]][as.numeric(lmin[2])],
                         e_hull[[1]][lmin_plus:length(e_hull[[1]])])
      }
    }
    e_hull[[2]] <- e_hull[[2]][!(e_hull[[2]] %in% e_hull[[2]][as.numeric(lmin[2])])]
    hull_plot(e_hull)
    print(paste(e_hull))
    invisible(readline(prompt="Press [enter] to continue"))
  }  
  e_hull[[2]] <- e_hull[[1]]
  e_hull <- e_hull[2:length(e_hull)]
}

hull_plot(e_hull)
print(sum(sapply(1:(nrow(mx)-1), 
                 FUN = function(X){mx[e_hull[[1]][X],e_hull[[1]][X+1]]}), 
          mx[e_hull[[1]][1], e_hull[[1]][length(e_hull[[1]])]]))