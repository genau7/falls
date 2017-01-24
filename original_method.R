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



# ======== loess ===================
mag.JW.s1 <- noisy.mag.JW.s1
for (i in 1:36){
  sample <- noisy.mag.JW.s1[i,]
  mag.JW.s1[i,] <- smooth.loess(sample, 0.1)
  comparePlots(noisy.mag.JW.s1, mag.JW.s1, i)
}

mc.pos.JW.sw1 <- noisy.mc.pos.JW.s1
mc.vel.JW.sw1 <- noisy.mc.vel.JW.s1
mc.acc.JW.sw1 <- noisy.mc.acc.JW.s1
for (scenario in 1:36){
  for (dim in 1:3){
    sample <- noisy.mc.pos.JW.s1[scenario, dim,]
    plotScenario(noisy.mc.pos.JW.s1, scenario, dim)
    derivatives <- smooth.loess.deriv(sample)
    mc.pos.JW.sw1[scenario, dim,] <- derivatives$pos
    mc.vel.JW.sw1[scenario, dim,] <- derivatives$vel
    mc.acc.JW.sw1[scenario, dim,] <- derivatives$acc
    #comparePlots(noisy.mc.pos.JW.s1[,dim,], mc.pos.JW.sw1[, dim,], scenario)
    plotScenario(mc.vel.JW.s1, i, dims["z"])
    plot(derivatives$vel, type='l')
  }
}

toplot <- mc.acc.JW.sw1
for (scenario in 1:36){
  scenario <- 1
   x <- toplot[scenario, 1,]
   y <- toplot[scenario, 2,]
   z <- toplot[scenario, 3,]
   plot(z, type='l', col='red', ylim=c(-0.02, 0.02))
   lines(x, col='green')
   lines(y, col="blue")
}
plot(mc.vel.JW.s1[1,3,], type='l')


z <- mc.pos.JW.sw1[1,3,]
x <- mc.pos.JW.sw1[1,1,]
y <- mc.pos.JW.sw1[1,2,]
scatterplot3d(x, y, z, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="scatterplot3d - 1", type='l')

plot(y~x, type='l')
plot(z~y, type='l')
plot(z~x, type='l')

#calc six features: 36x6 features
features.all = data.frame(matrix(0, ncol=6, nrow=36,
                       dimnames=list(c(), c("maxVelXY", "maxVelZ", "maxAcc", "maxAccZ", "mag10", "z10"))))
#max velocity
features.all$maxVelXY[1] = max(mc.vel.JW.sw1[1,dims["x"],], mc.vel.JW.sw1[1,dims["y"],])
#max absolute velocity along Z axis

#max acceleration

#max absolute acceleratio along Z axis

#magnitude in the 10th sample after max acceleration

#z position in the 10th sample after max acceleration





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

