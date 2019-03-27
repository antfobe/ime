if(!require("install.load")) {install.packages("install.load"); library("install.load")}
invisible(
  lapply(c('data.table', 'vars', 'fasttime', 'RSQLite', 'sqldf'), 
         FUN = install_load));

tgs <- fread(file = "~/datasets/march_week3_csv/blacklist_flows_cut.csv", header = FALSE)
tgs <- rbind(tgs, fread(file = "~/datasets/march_week3_csv/spam_flows_cut.csv", header = FALSE))
tgs <- rbind(tgs, fread(file = "~/datasets/march_week3_csv/sshscan_flows_cut.csv", header = FALSE))
tgs <- unique(tgs$V4)
tgsdf <- data.frame(matrix(unlist(tgs), nrow=length(tgs), byrow=TRUE),stringsAsFactors=FALSE)

con <- dbConnect(SQLite(), "~/datasets/march_week3_csv/march_week3.sqlite")
message("creating table for csv data")
read.csv.sql("~/datasets/march_week3/march.week3.csv.uniqblacklistremoved", sql = "CREATE TABLE march_week3 AS SELECT * FROM file;")
dbWriteTable(con, "mar_w3_victs", tgsdf, overwrite = TRUE)

dbSendQuery(con, "ALTER TABLE march_week3 ADD COLUMN victim INTEGER;")
dbSendQuery(con, "UPDATE march_week3 SET victim = 0 WHERE da NOT IN (SELECT victims FROM mar_w3_victs);")
dbSendQuery(con, "UPDATE march_week3 SET victim = 1 WHERE da IN (SELECT victims FROM mar_w3_victs);")

tgdf <- dbFetch(dbSendQuery(con, "SELECT te, count(da), sum(byt) FROM march_week3 WHERE victim == 0 GROUP BY te"))
names(tgdf) <- c("te", "#da", "sum_byt")
tgdf$avg_byt <- tgdf$sum_byt / tgdf$"#da"
fwrite(tgdf, "~/datasets/march_week3_csv/ntg_byts.csv", append=FALSE)

tgdf <- dbFetch(dbSendQuery(con, "SELECT te, count(da), sum(byt) FROM march_week3 WHERE victim == 1 GROUP BY te"))
names(tgdf) <- c("te", "#da", "sum_byt")
tgdf$avg_byt <- tgdf$sum_byt / tgdf$"#da"
fwrite(tgdf, "~/datasets/march_week3_csv/tg_byts.csv", append=FALSE)
dbDisconect(con)

