library(R.matlab)
library(mlbench)
require(nnet)
source("utils.R")

dims <- c(1,2,3)
names(dims) <- c("x", "y", "z")

#pkgs <- c('doParallel', 'foreach')
#lapply(pkgs, require, character.only = T)
#registerDoParallel(cores = 4)

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

mc.pos.JW.s1 <- noisy.mc.pos.JW.s1
mc.vel.JW.s1 <- noisy.mc.vel.JW.s1
mc.acc.JW.s1 <- noisy.mc.acc.JW.s1
for (dim in 1:3){
  mc.pos.JW.s1[,dim,] <- smooth.nn(noisy.mc.pos.JW.s1[,dim,])
  mc.vel.JW.s1[,dim,] <- calcDerivative(mc.pos.JW.s1[,dim,])
  mc.acc.JW.s1[,dim,] <- calcDerivative(mc.vel.JW.s1[,dim,])
}


toplot <- mc.vel.JW.s1
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
z <- mc.pos.JW.s1[1,3,]
x <- mc.pos.JW.s1[1,1,]
y <- mc.pos.JW.s1[1,2,]
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
  (index <- which.max(abs(mc.acc.JW.s1[scenario, dims["z"],])))
  
  #max velocity in XY plane
  features.all$maxVelXY[scenario] = max(mc.vel.JW.s1[scenario,dims["x"],], mc.vel.JW.s1[scenario,dims["y"],])
  
  #max absolute velocity along Z axis
  features.all$maxVelZ[scenario] = max(abs(mc.vel.JW.s1[scenario,dims["z"],]))
  
  #max acceleration in XY plane
  features.all$maxAccXY[scenario] = max(mc.acc.JW.s1[scenario, dims["x"],], mc.acc.JW.s1[scenario, dims["y"],])

  #max absolute acceleration along Z axis
  features.all$maxAccZ[scenario] = abs(mc.acc.JW.s1[scenario, dims["z"],index])
  
  if(index > 200)
    index=200
  
  #magnitude in the 10th sample after max z acceleration
  features.all$mag10[scenario] <- mag.JW.s1[scenario, index+10]
  
  #z position in the 10th sample after max z acceleration
  features.all$z10[scenario] <- mc.pos.JW.s1[scenario, dims["z"], index+10]
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

library(SDMTools)
preds <- loo(scaled, train_, test_)
cm <- confusion.matrix(features.all$fall, preds, threshold = 0.5)


c# ============= PCA ================================
pca <- princomp(features, cor=TRUE)
summary(pca)
loadings(pca)
plot(pca)
biplot(pca)
pred <- predict(pca)
features.pca <- data.frame(matrix(0, ncol=2, nrow=36,
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

preds <- loo(scaled, train_, test_)
cm <- confusion.matrix(features.all$fall, preds, threshold = 0.5)




















# ============== NPCA =========================

library(homals)
NPCA <- homals(data=data, ndim = 2, active = TRUE, level = "numerical")
pred <- predict(NPCA)
library(rgl)
plot(NPCA)
biplot(pred)
plot3dstatic(NPCA)
plot(NPCA, plot.dim=c(1:2), plot.type="graphplot")

res <- homals(galo, active=c(TRUE, TRUE,TRUE, TRUE, FALSE))
pr.res <- predict(res)
pr.res


#source("https://bioconductor.org/biocLite.R")
biocLite("pcaMethods")
library(pcaMethods)
Matrix <- as.matrix(scaled[,-4])
Matrix <- as.matrix(data)

#npca <- pca(data, nPcs=2, method="nlpca", maxSteps=2 * prod(dim(Matrix))) #mean scaled
lala <- nlpca(Matrix, nPcs = 2, maxSteps = 2 * prod(dim(Matrix)), unitsPerLayer = NULL, 
      functionsPerLayer = NULL, weightDecay = 0.001, weights = NULL)

(predict(lala, data))
pred <- fitted(lala, Matrix)
plot(pred[,1], Matrix[,1])
slplot(lala)

head(pred)
head(Matrix)



data(helix)
helixNA <- helix
## not a single complete observation
helixNA <- t(apply(helix, 1, function(x) { x[sample(1:3, 1)] <- NA; x}))
## 50 steps is not enough, for good estimation use 1000
helixNlPca <- pca(helixNA, nPcs=1, method="nlpca", maxSteps=50)
fittedData <- fitted(helixNlPca, helixNA)
plot(fittedData[which(is.na(helixNA))], helix[which(is.na(helixNA))])
## compared to solution by Nipals PCA which cannot extract non-linear patterns
helixNipPca <- pca(helixNA, nPcs=2)
fittedData <- fitted(helixNipPca)
plot(fittedData[which(is.na(helixNA))], helix[which(is.na(helixNA))])
