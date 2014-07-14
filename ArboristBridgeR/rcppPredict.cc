# Copyright (C)  2012-2014   Mark Seligman
##
## This file is part of ArboristBridgeR.
##
## ArboristBridgeR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## ArboristBridgeR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

#include <Rcpp.h>

using namespace Rcpp;
using namespace std;

#include "predict.h"

// Prediction on out-of-box test set.
//
// Returns value of mean-square error in 'sError'.  Writes the per-predictor Gini values.
//
// Individual predictions are not exposed.  Rather, the OOB/test/importance statistics
// are averaged and returned.
//
RcppExport SEXP RcppPredictOOBQuant(SEXP sPredGini, SEXP sError, SEXP sQuantVec, SEXP sQPred) {
  NumericVector error(sError);
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector predGini(sPredGini);

  Predict::PredictOOBQuant(error.begin(), quantVec.begin(), quantVec.length(), qPred.begin(), predGini.begin());

  return wrap(0);
}

RcppExport SEXP RcppPredictOOB(SEXP sPredGini, SEXP sError) {
  NumericVector error(sError);
  NumericVector predGini(sPredGini);

  Predict::PredictOOBReg(error.begin(), predGini.begin());

  return wrap(0);
}


RcppExport SEXP RcppPredictOOBCtg(SEXP sPredGini, SEXP sConf, SEXP sError) {
  IntegerMatrix conf(sConf);
  NumericVector error(sError);
  NumericVector predGini(sPredGini);

  Predict::PredictOOBCtg(conf.begin(), error.begin(), predGini.begin());

  return wrap(0);
}

// The only clients are the Predict variants, hence the inclusion of the
// reload method in this file.
//
RcppExport SEXP RcppReload(SEXP sPreds, SEXP sSplits, SEXP sScores, SEXP sBumpL, SEXP sBumpR, SEXP sOrigins, SEXP sFacOff, SEXP sFacSplits) {
  IntegerVector preds(sPreds);
  NumericVector splits(sSplits);
  NumericVector scores(sScores);
  IntegerVector bumpL(sBumpL);
  IntegerVector bumpR(sBumpR);
  IntegerVector origins(sOrigins);
  IntegerVector facOff(sFacOff);
  IntegerVector facSplits(sFacSplits);

  if (facSplits.length() == 0)
    Predict::ForestReload(origins.length(), preds.length(), preds.begin(), splits.begin(), scores.begin(), bumpL.begin(), bumpR.begin(), origins.begin(), 0, 0);
  else
    Predict::ForestReload(origins.length(), preds.length(), preds.begin(), splits.begin(), scores.begin(), bumpL.begin(), bumpR.begin(), origins.begin(), facOff.begin(), facSplits.begin());
}

RcppExport SEXP RcppReloadQuant(SEXP sQYRanked, SEXP sQRankOrigin, SEXP sQRank, SEXP sQRankCount, SEXP sQLeafPos, SEXP sQLeafExtent) {
  NumericVector qYRanked(sQYRanked);
  IntegerVector qRankOrigin(sQRankOrigin);
  IntegerVector qRank(sQRank);
  IntegerVector qRankCount(sQRankCount);
  IntegerVector qLeafPos(sQLeafPos);
  IntegerVector qLeafExtent(sQLeafExtent);

  Predict::ForestReloadQuant(qYRanked.begin(), qYRanked.length(), qRankOrigin.begin(), qRank.begin(), qRankCount.begin(), qLeafPos.begin(), qLeafExtent.begin());
}

RcppExport SEXP RcppPredictQuant(SEXP sQuantVec, SEXP sQPred, SEXP sY) {
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector y(sY);

  Predict::PredictQuant(quantVec.begin(), quantVec.length(), qPred.begin(), y.begin());
}

RcppExport SEXP RcppPredictReg(SEXP sy) {
  NumericVector y(sy);

  Predict::PredictReg(y.begin());

  return wrap(0);
}

// 
//
RcppExport SEXP RcppPredictCtg(SEXP sy, SEXP sCtgWidth) {
  IntegerVector y(sy);
  int ctgWidth = as<int>(sCtgWidth);
  Predict::PredictCtg(y.begin(), ctgWidth);
  y = y + 1;

  return wrap(0);
}

