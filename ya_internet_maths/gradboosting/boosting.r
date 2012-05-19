# @author: Alexandra Mikhaylova mikhaylova.alexandra.a@gmail.com

# install.packages("/home/mikhaylova/R/gbm_1.6-3.2.tar.gz", "/home/mikhaylova/R", repos=NULL)
learn <- read.table("imat2009_learning.dat", sep="\t")
iter <- 1000
library(gbm, lib.loc="/home/mikhaylova/R")
print("Package loaded, work started")
model <- gbm(V1~., data=learn, distribution="gaussian", n.trees=iter, shrinkage=0.015,
    interaction.depth=8, n.minobsinnode=10)
write.table(predict.gbm(model, read.table("imat2009_test.dat"), iter), "imat2009_test_result.dat", row.names=FALSE, col.names=FALSE)
