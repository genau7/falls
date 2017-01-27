library(R.matlab)
library(mlbench)
require(nnet)
source("utils.R")

dims <- c(1,2,3)
names(dims) <- c("x", "y", "z")

#pkgs <- c('doParallel', 'foreach')
#lapply(pkgs, require, character.only = T)
#registerDoParallel(cores = 4)

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
mc.vel.JW.sw1 <- noisy.mc.vel.JW.s1
mc.acc.JW.sw1 <- noisy.mc.acc.JW.s1
for (dim in 1:3){
  mc.pos.JW.sw1[,dim,] <- smooth.nn(noisy.mc.pos.JW.s1[,dim,])
  mc.vel.JW.sw1[,dim,] <- calcDerivative(mc.pos.JW.sw1[,dim,])
  mc.acc.JW.sw1[,dim,] <- calcDerivative(mc.vel.JW.sw1[,dim,])
}


toplot <- mc.vel.JW.sw1
for (scenario in 1:36){
  scenario <- 1
  x <- toplot[scenario, 1,]
  y <- toplot[scenario, 2,]
  z <- toplot[scenario, 3,]
  plot(z, type='l', col='red', ylim=c(-0.2, 0.2))
  lines(x, col='green')
  lines(y, col="blue")
}


library(scatterplot3d)
z <- mc.pos.JW.sw1[1,3,]
x <- mc.pos.JW.sw1[1,1,]
y <- mc.pos.JW.sw1[1,2,]
scatterplot3d(x, y, z, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="scatterplot3d - 1", type='l')

plot(y~x, type='l')
plot(z~y, type='l')
plot(z~x, type='l')

#calc six features: 36x6 feature
features.all = data.frame(matrix(0, ncol=7, nrow=36,
                       dimnames=list(c(), c("maxVelXY", "maxVelZ", "maxAccXY", "maxAccZ", "mag10", "z10", "fall"))))


for(scenario in 1:36){
  #find index of the biggest absolute acceleration along Z axis
  (index <- which.max(abs(mc.acc.JW.sw1[scenario, dims["z"],])))
  
  #max velocity in XY plane
  features.all$maxVelXY[scenario] = max(mc.vel.JW.sw1[scenario,dims["x"],], mc.vel.JW.sw1[scenario,dims["y"],])
  
  #max absolute velocity along Z axis
  features.all$maxVelZ[scenario] = max(abs(mc.vel.JW.sw1[scenario,dims["z"],]))
  
  #max acceleration in XY plane
  features.all$maxAccXY[scenario] = max(mc.acc.JW.sw1[scenario, dims["x"],], mc.acc.JW.sw1[scenario, dims["y"],])

  #max absolute acceleration along Z axis
  features.all$maxAccZ[scenario] = abs(mc.acc.JW.sw1[scenario, dims["z"],index])
  
  if(index > 200)
    index=200
  
  #magnitude in the 10th sample after max z acceleration
  features.all$mag10[scenario] <- mag.JW.s1[scenario, index+10]
  
  #z position in the 10th sample after max z acceleration
  features.all$z10[scenario] <- mc.pos.JW.sw1[scenario, dims["z"], index+10]
}
features.all$fall[1:18] <- 1
features.all$fall[19:36] <- 0
features.all$fall <- as.factor(features.all$fall)

library(caret)
library(mlbench)
set.seed(7)
features <- features.all

# == Remove variables with high absolute correlation
(correlationMatrix <- cor(features[,-7]))
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(colnames(features)[highlyCorrelated])
features <- features[, -highlyCorrelated]

# == Find the most important feature () using an LVQ model

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(features$fall~., data=features, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


# ==== ????????? ============================================

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(features[,-6], features$fall, sizes=c(1:5), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


# ==== missing part about choosing mag10????? =============


# =================== NN classification ========================================================================
features <- features.all[, c(4,5,6)]


#normalize features

scaled <- as.data.frame(scale(features, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall
index <- sample(1:nrow(features),round(0.75*nrow(features)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)
n <- names(train_)
f <- as.formula(paste("fall ~", paste(n[!n %in% "fall"], collapse = " + ")))
nnet.fit <- nnet(fall~., data=train_, size=20, decay=5e-4, maxit=200)
(predict(nnet.fit, test_)>0.5)


selectedFeatures <- c(4,5,6)
data <- features.all[, selectedFeatures]
maxs <- apply(data, 2, max)
maxs <- sapply(maxs, as.numeric)
mins <- apply(data, 2, min)
mins <- sapply(mins, as.numeric)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall


loo(data, train_, test_)

# leave one out validation
for(i in 1:nrow(features)){
  i <- 1
  train_ <- scaled[-i,]
  test_ <- scaled[i,]
  nnet.fit <- nnet(fall~., data=train_, size=20, decay=5e-4, maxit=200)
  (predict(nnet.fit, test_)>0.5)
}

