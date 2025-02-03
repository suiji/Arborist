library(Rborist)
context("Sampling, replacement modes")

sampleSize <- function(nObs, nRep, withRepl) {
    s <- runif(nObs)
    ps <- presample(s, nRep=nRep, withRepl=withRepl)
    length(ps$samples)
}

sampleSizePass <- function(nObs, nRep, withRepl, sizeEst, tol) {
  nSamp <- sampleSize(nObs, nRep, withRepl)
  ratio <- nSamp / sizeEst
  if ((ratio >= 1.0 - tol) && (ratio <= 1.0 + tol)) {
    pass <- 1
  }
  else {
    pass <- 0
  }
    
  pass
}


test_that("Sampling, sample sizes", {
    expect_equal( sampleSizePass(nObs=1000, nRep=100, withRepl = TRUE, sizeEst=63000, tol=0.05), 1)
    expect_equal( sampleSizePass(nObs=1000, nRep=100, withRepl = FALSE, sizeEst=63000, tol=0.05), 1)
})