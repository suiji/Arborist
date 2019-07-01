library(Rborist)
context("Regression, numeric predictors")

test_that("Numeric-only regression accuracy", {
  testthat::skip_on_cran()
  expect_equal( regNumPass(10, 1000, 20), 1)
})

regNumPass <-function(m, nrow, ncol) {
  y<-numeric(nrow)
  a<-runif(ncol)
  b<-runif(ncol)*m
  x<-matrix(runif(nrow*ncol)*m,nrow,ncol)
  for(i in 1:nrow){
    for(j in 1:ncol){
      y[i]<-y[i]+b[j]*round(x[i,j]/m-a[j]+0.5)
    }
  }

  rs <- Rborist(x, y, nTree = 500)
  pass <- ifelse(rs$rsq >= 0.7, 1, 0)
}
