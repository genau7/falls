library(R.matlab)
library(mlbench)
require(nnet)
source("utils.R")
set.seed(127)

#pkgs <- c('doParallel', 'foreach')
#lapply(pkgs, require, character.only = T)
#registerDoParallel(cores = 4)

smoothData <- function(mtarrays){
  noisy.mag <- mtarrays[['mag']]
  noisy.mc.pos <- mtarrays[['mc.pos']]
  noisy.mc.vel <- mtarrays[['mc.vel']]
  noisy.mc.acc <- mtarrays[['mc.acc']]
  
  
  #filter magnitude and pos using NN
  mag <- smooth.nn(noisy.mag)

  mc.pos <- noisy.mc.pos
  mc.vel <- noisy.mc.vel
  mc.acc <- noisy.mc.acc
  for (dim in 1:3){
    mc.pos[,dim,] <- smooth.nn(noisy.mc.pos[,dim,])
    mc.vel[,dim,] <- calcDerivative(mc.pos[,dim,])
    mc.acc[,dim,] <- calcDerivative(mc.vel[,dim,])
  }
  return(list("mag"=mag, "pos"=mc.pos, "vel"=mc.vel, "acc"=mc.acc))
}

#load data from actor JW from sensor s1
mtarrays <- readMat('../matlab_dane/mc_data_JW_s1.mat', fixNames=TRUE)
jw.sensor1 <- smoothData(mtarrays)
mag <- jw.sensor1$mag
mc.pos <- jw.sensor1$pos
mc.vel <- jw.sensor1$vel
mc.acc <- jw.sensor1$acc

library(abind)
mtarrays <- readMat('../matlab_dane/mc_data_JW_s2.mat', fixNames=TRUE)
jw.sensor2 <- smoothData(mtarrays)
mag <- rbind(mag, jw.sensor2$mag)
mc.pos <- abind(mc.pos, jw.sensor2$pos, along = 1)
mc.vel <- abind(mc.vel, jw.sensor2$vel, along = 1)
mc.acc <- abind(mc.acc, jw.sensor2$acc, along = 1)

mtarrays <- readMat('../matlab_dane/mc_data_PM_s1.mat', fixNames=TRUE)
pm.sensor1 <- smoothData(mtarrays)
mag <- rbind(mag, pm.sensor1$mag)
mc.pos <- abind(mc.pos, pm.sensor1$pos, along = 1)
mc.vel <- abind(mc.vel, pm.sensor1$vel, along = 1)
mc.acc <- abind(mc.acc, pm.sensor1$acc, along = 1)

mtarrays <- readMat('../matlab_dane/mc_data_PM_s2.mat', fixNames=TRUE)
pm.sensor2 <- smoothData(mtarrays)
mag <- rbind(mag, pm.sensor2$mag)
mc.pos <- abind(mc.pos, pm.sensor2$pos, along = 1)
mc.vel <- abind(mc.vel, pm.sensor2$vel, along = 1)
mc.acc <- abind(mc.acc, pm.sensor2$acc, along = 1)

toplot <- mc.acc
   for (scenario in 1:36){
     scenario <- 1
     x <- toplot[scenario, 1,]
     y <- toplot[scenario, 2,]
     z <- toplot[scenario, 3,]
     plot(z, type='l', col=0, ylim=c(-0.02, 0.02))
     lines(x, col='green')
     lines(y, col="blue")
}



#calc six features: 144x6 feature
scenariosNum <- nrow(mag)
features.all = data.frame(matrix(0, ncol=7, nrow=scenariosNum,
                       dimnames=list(c(), c("maxVelXY", "maxVelZ", "maxAccXY", "maxAccZ", "mag10", "z10", "fall"))))


for(scenario in 1:scenariosNum){
  #find index of the biggest absolute acceleration along Z axis
  (index <- which.max(abs(mc.acc[scenario, dims["z"],])))
  
  #max velocity in XY plane
  features.all$maxVelXY[scenario] = max(mc.vel[scenario,dims["x"],], mc.vel[scenario,dims["y"],])
  
  #max absolute velocity along Z axis
  features.all$maxVelZ[scenario] = max(abs(mc.vel[scenario,dims["z"],]))
  
  #max acceleration in XY plane
  features.all$maxAccXY[scenario] = max(mc.acc[scenario, dims["x"],], mc.acc[scenario, dims["y"],])

  #max absolute acceleration along Z axis
  features.all$maxAccZ[scenario] = abs(mc.acc[scenario, dims["z"],index])
  
  if(index > 200)
    index=200
  
  #magnitude in the 10th sample after max z acceleration
  features.all$mag10[scenario] <- mag[scenario, index+10]
  
  #z position in the 10th sample after max z acceleration
  features.all$z10[scenario] <- mc.pos[scenario, dims["z"], index+10]
}

falls <- c(1:18, 37:54, 73:90, 109:126)
features.all$fall[falls] <- 1
features.all$fall <- as.factor(features.all$fall)


library(scatterplot3d)
z <- mc.pos[1,3,]
x <- mc.pos[1,1,]
y <- mc.pos[1,2,]
scatterplot3d(x, y, z, highlight.3d=TRUE, col.axis="blue",
              col.grid="lightblue", main="scatterplot3d - 1", type='l')

plot(y~x, type='l', )
plot(z~y, type='l')
plot(z~x, type='l')



library(caret)
library(mlbench)
features <- features.all

# == Remove variables with high absolute correlation
(correlationMatrix <- cor(features[,-7]))
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(colnames(features)[highlyCorrelated])
features <- features[, -highlyCorrelated]


# ==== First NN classifier with 4 variables ===========
#normalize features
selectedFeatures <- c(1,4,5,6)
data <- features.all[, selectedFeatures]
maxs <- apply(data, 2, max)
maxs <- sapply(maxs, as.numeric)
mins <- apply(data, 2, min)
mins <- sapply(mins, as.numeric)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall

library(SDMTools)
preds <- loo(scaled)
(cm.4V <- confusion.matrix(features.all$fall, preds, threshold = 0.5))



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
results <- rfe(features[,-5], features$fall, sizes=c(1:5), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


# ==== missing part about choosing mag10????? =============


# =================== NN classification for 3 vars ========================================================================
#normalize features
selectedFeatures <- c(4,5,6)
data <- features.all[, selectedFeatures]
maxs <- apply(data, 2, max)
maxs <- sapply(maxs, as.numeric)
mins <- apply(data, 2, min)
mins <- sapply(mins, as.numeric)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall


preds <- loo(scaled)
(cm.3V <- confusion.matrix(features.all$fall, preds, threshold = 0.5))


# ============= PCA 3 => 2 ================================
features <- data
pca <- princomp(features, cor=TRUE)
summary(pca)
loadings(pca)
plot(pca)
biplot(pca)
pred <- predict(pca)
features.pca <- data.frame(matrix(0, ncol=2, nrow=scenariosNum,
                                  dimnames=list(c(), c("pc1", "pc2"))))

features.pca$pc1 <- pred[,1]
features.pca$pc2 <- pred[,2]

# ============= PCA 4 => 2 ================================
selectedFeatures <- c(1,4,5,6)
features <- features.all[selectedFeatures]
pca <- princomp(features, cor=TRUE)
summary(pca)
loadings(pca)
plot(pca)
biplot(pca)
pred <- predict(pca)
features.pca <- data.frame(matrix(0, ncol=2, nrow=scenariosNum,
                                  dimnames=list(c(), c("pc1", "pc2"))))

features.pca$pc1 <- pred[,1]
features.pca$pc2 <- pred[,2]


# ============ classifier after PCA ================

data <- features.pca
maxs <- apply(data, 2, max)
maxs <- sapply(maxs, as.numeric)
mins <- apply(data, 2, min)
mins <- sapply(mins, as.numeric)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall

preds <- loo(scaled)
(cm.pca4 <- confusion.matrix(features.all$fall, preds, threshold = 0.5))

# =========== KPCA 3 => 2 ================================
selectedFeatures <- c(4,5,6)
features <- features.all[selectedFeatures]
kpc <- kpca(~.,data=features, kernel="rbfdot", kpar=list(sigma=0.2), features=2)
pred <- predict(kpc, features)
plot(rotated(kpc),col=as.integer(features.all$fall),
     xlab="1st Principal Component",ylab="2nd Principal Component")

features.kpca <- data.frame(matrix(0, ncol=2, nrow=scenariosNum,
                                  dimnames=list(c(), c("kpc1", "kpc2"))))

features.kpca$kpc1 <- pred[,1]
features.kpca$kpc2 <- pred[,2]


data <- features.kpca
maxs <- apply(data, 2, max)
maxs <- sapply(maxs, as.numeric)
mins <- apply(data, 2, min)
mins <- sapply(mins, as.numeric)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$fall <- features.all$fall

preds <- loo(scaled)
(cm.kpca3 <- confusion.matrix(features.all$fall, preds, threshold = 0.5))


