if(!require("install.load")) {install.packages("install.load"); library("install.load")}
invisible(
  lapply(c('data.table', 'vars', 'fasttime'), 
         FUN = install_load));

for (f in list.files(path = "/opt/large/", pattern = "*week4_part*", full.names = TRUE)) {
    print(paste("doing file ", f, sep = "-> "))
  hfd <- fread(file = f, header = FALSE)
  names(hfd) <- c("te", "td", "sa", "da", "sp", "dp", "pr", "flg", "fwd", 
                  "stos", "pkt", "byt", "tag")
  setDT(hfd)[, "te" := lapply(.SD, fasttime::fastPOSIXct), .SDcols = "te"]
  hfd[, c("sa", "da")] <- setDT(hfd)[, freq := .N, by = .(sa, da)][order(-freq), .(sa, da)]
  setDT(hfd)[, c("sa", "da") := lapply(.SD, FUN = match, table = unique(c(hfd$sa, hfd$da))), 
             .SDcols = c("sa", "da")]
  hfd[, c("sp", "dp")] <- setDT(hfd)[, freq := .N, by = .(sp, dp)][order(-freq), .(sp, dp)]
  setDT(hfd)[, c("sp", "dp") := lapply(.SD, FUN = match, table = unique(c(hfd$sp, hfd$dp))), 
             .SDcols = c("sp", "dp")]
  setDT(hfd)[, "pr" := lapply(.SD, FUN = match, table = unique(hfd$pr)), .SDcols = "pr"]
  setDT(hfd)[, "flg" := lapply(.SD, FUN = match, table = unique(hfd$flg)), .SDcols = "flg"]
  setDT(hfd)[, "tag" := lapply(.SD, FUN = match, table = unique(hfd$tag)), .SDcols = "tag"]
    print(paste("done conversions ", f, sep = "-> "))
    # VAR(hfd[, c("td", "sa", "pr", "flg", "tag", "stos", "pkt")][1:1000, ], p = 1, type = "trend")
  VARmdl <- VAR(hfd[, c(2,3,4,5,6,7,8,10,11,13)][1:(nrow(hfd) %/% 10), ], 
                p = 6, type = "both")
  saveRDS(object = VARmdl, file = paste("/opt/large/mw4",
                                        regmatches(f, regexpr(pattern = "_part.*", f, perl = TRUE)),
                                        ".rds"))
    rm(hfd, VARmdl)
}

# causality(VAR(hfd[1:100, c(1,2,3,4,5,6,7,8,10,11,13)], type = "both"))
