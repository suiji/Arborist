library(Rborist)
context("Regression, numeric predictors")

regNumPass <-function(m, nrow, ncol, minRsq) {
  y<-numeric(nrow)
  a<-runif(ncol)
  b<-runif(ncol)*m
  x<-matrix(runif(nrow*ncol)*m,nrow,ncol)
  for(i in 1:nrow){
    for(j in 1:ncol){
      y[i]<-y[i]+b[j]*round(x[i,j]/m-a[j]+0.5)
    }
  }

  rs <- rfArb(x, y)
  if (rs$validation$rsq >= minRsq) {
      pass <- 1

  }
  else {
      pass <- 0
  }
  pass
}

test_that("Numeric-only regression accuracy", {
    skip() # Until intermittent thread-grabbing behaviour corrected.
    minRsq <- 0.6 # very lenient threshold for passing
    expect_equal( regNumPass(10, 1000, 20, minRsq), 1)
})
