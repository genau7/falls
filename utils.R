# ============================================================
#       utils for smoothing and plotting data
#       author: Katarzyna Stepek
# ============================================================

plotScenario <- function(data, scenario, dimension=0){
  if (dimension != 0){
    plot(data[scenario, dimension,], pch=".", main=toString(scenario))
    lines(data[scenario, dimension,])}
  else {
    plot(data[scenario,], pch=".", main=toString(scenario))
    lines(data[scenario,])}  
}

comparePlots <- function(data1, data2, scenario, dimension=0){
  if (dimension != 0){
    plot(data1[scenario, dimension,], pch=".", main=toString(scenario))
    lines(data1[scenario, dimension,])
    lines(data2[scenario, dimension,], col="blue")}
  else {
    plot(data1[scenario,], pch=".", main=toString(scenario))
    lines(data1[scenario,])
    lines(data2[scenario,], col="red")}  
}

# =========== Loess smoothing ===============================

# Smooth a single scenario
smooth.loess <- function(y, span=0.1){
  x <- 1:length(y)
  mymodel <- loess(y~x, span = span)
  return(predict(mymodel))
}


# =========== Neural network smoothing =======================

#takes in a sample of 300 frames and filters it using a neural net with one hidden layer with 10 neurons. 
#The NN input is scaled to <0,1> and the output scale10 back to regular scale
filterScenario <- function(sample){
  x <- seq(1/length(sample), 1, 1/length(sample))
  nnet.fit <- nnet(sample/max(sample)~x, size=15, linout=TRUE, trace=FALSE)
  return(predict(nnet.fit)*max(sample))
}

smooth.nn <- function(y){
  result <- apply(y, 1, filterScenario)
  result <- apply(result, 1 , identity)
  return(result)
}