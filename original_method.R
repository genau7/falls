library(R.matlab)
library(mlbench)
require(nnet)
source("utils.R")
dims <- c(1,2,3)
names(dims) <- c("x", "y", "z")

# ============= functions ==============================



# ============== eof functions ========================

#load data from actor JW from sensor s1
mtarrays <- readMat('../matlab_dane/mc_data_JW_s1.mat', fixNames=TRUE)

noisy.mag.JW.s1 <- mtarrays[['mag']]
noisy.mc.pos.JW.s1 <- mtarrays[['mc.pos']]
noisy.mc.vel.JW.s1 <- mtarrays[['mc.vel']]
noisy.mc.acc.JW.s1 <- mtarrays[['mc.acc']]


#filter magnitude and pos using NN
mag.JW.s1 <- smooth.nn(noisy.mag.JW.s1)
for (i in 1:36){
  # plotScenario(mag.JW.s1, i)
  comparePlots(noisy.mag.JW.s1, mag.JW.s1, i)
}


mc.pos.JW.sw1 <- noisy.mc.pos.JW.s1
for (i in 1:3){
  mc.pos.JW.sw1[,i,] <- smooth.nn(noisy.mc.pos.JW.s1[,i,])
}

for (i in 1:36){
  # plotScenario(noisy.mc.pos.JW.s1, i, dims["x"])
  comparePlots(noisy.mc.pos.JW.s1, mc.pos.JW.sw1, i, dims["x"])
}

plotScenario(mc.pos.JW.sw1, 1, 1)
comparePlots(noisy.mc.pos.JW.s1, mc.pos.JW.sw1, 1, dims["x"])

# ======== loess ===================

for (i in 1:36){
  sample <- noisy.mag.JW.s1[i,]
  mag.JW.s1[i,] <- smooth.loess(sample)
  comparePlots(noisy.mag.JW.s1, mag.JW.s1, i)
}

  

#recalculate vel and acc



#any var with VIF >5 can be dropped 
