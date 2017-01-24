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
mag.JW.s1 <- noisy.mag.JW.s1
for (i in 1:36){
  sample <- noisy.mag.JW.s1[i,]
  mag.JW.s1[i,] <- smooth.loess(sample)
  comparePlots(noisy.mag.JW.s1, mag.JW.s1, i)
}

mc.pos.JW.sw1 <- noisy.mc.pos.JW.s1
for (scenario in 1:36){
  for (dim in 1:3){
    sample <- noisy.mc.pos.JW.s1[scenario, dim,]
    mc.pos.JW.sw1[scenario, dim,] <- smooth.loess(sample)
    comparePlots(noisy.mc.pos.JW.s1[,dim,], mc.pos.JW.sw1[, dim,], scenario)
  }
}
  

x <- 1:300
y <- noisy.mc.pos.JW.s1[1,3,]
m <- loess(y~x, span=0.1)
h <- h.select(x,y,method="cv")
dY <- diff(m$fitted)/diff(x)
dY <- c(dY[1], dY)
dY.model <-loess(dY~x, span=0.1)
plot(dY.model$fitted, col="green", type='l', main="vel")

ddY <- diff(dY.model$fitted)/diff(x)
ddY <- c(ddY[1], ddY)
ddY.model <-loess(ddY~x, span=0.1)
plot(ddY.model$fitted, col="green", type='l', main="acc")

m <- locpoly(x, y, bandwidth = h)
m.prime <- locpoly(x, y, bandwidth = h, drv=2)
y.pred <- predict(m)
y.prime <- predict(m, deriv=1)

plot(y, type="l")
lines(m, col="red")
plot(noisy.mc.vel.JW.s1[1,1,]~x, type='l')
plot(m.prime, col="green", type='l')
lines(m.prime, col="green")

#recalculate vel and acc
ds <- diff(noisy.mc.pos.JW.s1[1,1,])/diff(x)
ds <- c(ds[1], ds)
dt <- rowMeans(embed(x))
plot(noisy.mc.vel.JW.s1[1,1,]~x, type='l')
lines(ds~dt, col="red")

temp <- diff(noisy.mc.pos.JW.s1[1,1,])
temp2 <- diff(x)
df <- data.frame(t=1:300, s=noisy.mc.pos.JW.s1[1,1,], ds=c(temp[1], temp), dt=c(temp2[1], temp2))
df[50:100,]
pl#any var with VIF >5 can be dropped 

library(fields)
y <- noisy.mc.pos.JW.s1[1,1,]
m <- loess(y~x, span=0.1)
m <- smooth.spline(y, x, df = 6.4)
pred <- predict(m)
pred2 <- predict(m, x, deriv=1)

plot(x, y, type='l')
lines(pred, col="red")

y.prime <- diff(y)/diff(x)
pred.prime <- predict(m, deriv=2)
plot(y.prime, type='l')
plot(pred.prime, col="blue")

plot(mc.pos.JW.sw1[1,1,], type="l")
