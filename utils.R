# ============================================================
#       utils for smoothing and plotting data
#       author: Katarzyna Stepek
# ============================================================

dims <- c(1,2,3)
names(dims) <- c("x", "y", "z")

plotScenario <- function(data, scenario, dimension=0){
  if (dimension != 0){
    plot(data[scenario, dimension,], type='l', main=toString(scenario))}
  else {
    plot(data[scenario,], type='l', main=toString(scenario))}
}

comparePlots <- function(data1, data2, scenario, dimension=0){
  if (dimension != 0){
    plot(data1[scenario, dimension,], type='l', main=toString(scenario))
    lines(data2[scenario, dimension,], col="blue")}
  else {
    plot(data1[scenario,], type='l', main=toString(scenario))
    lines(data2[scenario,], col="red")}  
}

# =========== Loess smoothing ===============================

# Smooth a single scenario
smooth.loess <- function(y, span=0.1){
  x <- 1:length(y)
  mymodel <- loess(y~x, span = span)
  return(predict(mymodel))
}

smooth.loess.deriv <- function(y, span=0.1){
  x <- 1:length(y)
  mymodel <- loess(y~x, span=span)
  
  dY <- diff(mymodel$fitted)/diff(x)
  dY <- c(dY[1], dY)
  dY.model <-loess(dY~x, span=span)
  
  ddY <- diff(dY.model$fitted)/diff(x)
  ddY <- c(ddY[1], ddY)
  ddY.model <-loess(ddY~x, span=0.1)
  
  return(list("pos"=mymodel$fitted, "vel"=dY.model$fitted, "acc"=ddY.model$fitted))
}


# =========== Neural network smoothing =======================

#takes in a sample of 300 frames and filters it using a neural net with one hidden layer with 10 neurons. 
#The NN input is scaled to <0,1> and the output scale10 back to regular scale
filterScenario <- function(sample){
  x <- seq(1/length(sample), 1, 1/length(sample))

  biggest <- max(sample)
  smallest <- min(sample)
  scaled <- (sample-smallest)/(biggest-smallest)
  
  nnet.fit <- nnet(scaled~x, size=20, linout=TRUE, trace=FALSE, maxit=1200)
  return(predict(nnet.fit)*(biggest-smallest)+smallest)
}

smooth.nn <- function(y){
  result <- apply(y, 1, filterScenario)
  result <- apply(result, 1 , identity)
  return(result)
}

calcDerivative <- function(y){
  dY <- apply(y, 1, diff) # / diff(x), but diff(x)=1 so it was ignored
  dY <- apply(dY, 1 , identity)
  return(cbind(dY[,1], dY))
}

# ========= Classifier =================================

loo <- function(scaled){
  sampleSize <- nrow(scaled)
  preds <- 1:sampleSize
  for(i in 1:sampleSize){
    train_ <- scaled[-i,]
    test_ <- scaled[i,]
    nnet.fit <- nnet(fall~., data=train_, size=20, decay=5e-4, maxit=200, trace=FALSE)
    preds[i] <- predict(nnet.fit, test_)
  }
  return(preds)
}
