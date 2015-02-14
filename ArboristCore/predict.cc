// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.cc

   @brief Methods for the front-end interface supporting prediction.

   @author Mark Seligman

*/

// Interface class for front end.
// Holds simulation-specific parameters of the data.
//

#include "predict.h"
#include "dectree.h"
#include "quant.h"
#include "response.h"

// Testing only:
//#include <iostream>
using namespace std;

/**
   @brief Thin interface for reloading trained forest.

   @return void.
 */

void Predict::ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bump[], int _origins[], int _facOff[], int _facSplits[]) {
  DecTree::ForestReload(_nTree, _forestSize, _preds, _splits, _scores, _bump, _origins, _facOff, _facSplits);
}

/**
   @brief Thin interface for reloading trained forest, with quantiles.

   @return void.
 */

void Predict::ForestReloadQuant(int nTree, double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]) {
  Quant::FactoryPredict(nTree, qYRanked, qYLen, qRankOrigin, qRank, qRankCount, qLeafPos, qLeafExtent);
}

/**
   @brief Predicts using a regression forest on out-of-bag data.

   @param err is an output parameter containing the mean-square error.

   @param predInfo is an output vector parameter reporting predictor Info values.

   @return void, with output parameters.
 */
void Predict::PredictOOBReg(double *err, double predInfo[]) {
  ResponseReg::PredictOOB(err, predInfo);
}

/**
   @brief As above, but with quantile predictions as well.

   @param err is an output parameter containing the mean-square error.

   @param quantVec is an ordered vector of desired quantiles.

   @param qCount is the length of the quantVec vector.

   @param qPred is the predicted quantile set.

   @param predInfo is an output vector parameter reporting predictor Info values.

   @return void, with output parameters.
 */
void Predict::PredictOOBQuant(double err[], double quantVec[], int qCount, double qPred[], double predInfo[]) {
  Quant::EntryPredict(quantVec, qCount, qPred);
  ResponseReg::PredictOOB(err, predInfo);
}

/**
   @brief Predicts from a quantile forest.

   @return void, with output vector parameters.
 */

void Predict::PredictQuant(int nRow, double quantVec[], int qCount, double qPred[], double y[]) {
  Quant::EntryPredict(quantVec, qCount, qPred, nRow);
  DecTree::PredictAcrossReg(y, false);
}

/**
   @brief Predicts from a regression forest.

   @param y outputs the predicted response.

   @return void, with output vector parameter.
 */
void Predict::PredictReg(double y[]) {
  DecTree::PredictAcrossReg(y, false);
}

/**
   @brief Predicts using a classification forest on out-of-bag data.

   @param conf outputs a confusion matrix.

   @param err outputs the mean-square error.

   @param predInfo outputs predictor Info values.

   @return void, with output parameters.
 */
void Predict::PredictOOBCtg(int conf[], double *error, double predInfo[]) {
  ResponseCtg::PredictOOB(conf, error, predInfo);
}

/**
   @brief Predicts from a classification forest.

   @param y outputs the predicted response.

   @param ctgWidth is the cardinality of factor-valued response.

   @return void, with output parameter vector.
 */
void Predict::PredictCtg(int y[], int ctgWidth) {
  DecTree::PredictAcrossCtg(y, ctgWidth, 0, 0, false);
}
