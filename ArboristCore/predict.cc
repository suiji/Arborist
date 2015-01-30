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
#include "predictor.h"
#include "dectree.h"
#include "response.h"

#include <iostream>
using namespace std;

int QuantSig::qCells = 0;
double *QuantSig::qVec = 0;
double *QuantSig::qPred = 0;

/**
   @brief Sets the global parameters for quantile prediction using storage provided by the front end.

   @param _qVec

   @param _qCells

   @param _qPred

   @return void.
 */
void QuantSig::Factory(double *_qVec, int _qCells, double *_qPred) {
  qCells = _qCells;
  qVec = _qVec;
  qPred = _qPred;
}

/**
   @brief Unsets global state.

   @return void.

 */
void QuantSig::DeFactory() {
  qCells = 0;
  qVec = qPred = 0;
}

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

void Predict::ForestReloadQuant(double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]) {
  DecTree::ForestReloadQuant(qYRanked, qYLen, qRankOrigin, qRank, qRankCount, qLeafPos, qLeafExtent);
}

/**
   @brief Outputs Gini values of predictors and cleans up.

   @param predGini is an output vector of predictor Gini values.

   @return void, with output vector parameter.
 */
void Predict::Finish(double predGini[]) {
  DecTree::ScaleGini(predGini);
  Response::DeFactorySt();
}

void Predict::PredictOOBQuant(double *err, double *quantVec, int qCells, double qPred[], double predGini[]) {
  QuantSig::Factory(quantVec, qCells, qPred);
  Response::response->PredictOOB(0, err);
  QuantSig::DeFactory();
  Finish(predGini);
}

/**
   @brief Predicts using a regression forest on out-of-bag data.

   @param err is an output parameter containing the mean-square error.

   @param predGini is an output vector parameter reporting predictor Gini values.

   @return void, with output parameters.
 */
void Predict::PredictOOBReg(double *err, double predGini[]) {
  Response::response->PredictOOB(0, err);
  Finish(predGini);
}

/**
   @brief Predicts using a classification forest on out-of-bag data.

   @param conf outputs a confusion matrix.

   @param err outputs the mean-square error.

   @param predGini outputs predictor Gini values.

   @return void, with output parameters.
 */
void Predict::PredictOOBCtg(int conf[], double *error, double predGini[]) {
  Response::response->PredictOOB(conf, error);
  Finish(predGini);
}

/**
   @brief Predicts from a quantile forest.

   @return void, with output vector parameters.
 */

void Predict::PredictQuant(double quantVec[], int qCells, double qPred[], double y[]) {
  QuantSig::Factory(quantVec, qCells, qPred);
  DecTree::PredictAcrossReg(y, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}

/**
   @brief Predicts from a regression forest.

   @param y outputs the predicted response.

   @return void, with output vector parameter.
 */
void Predict::PredictReg(double y[]) {
  DecTree::PredictAcrossReg(y, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}

/**
   @brief Predicts from a classification forest.

   @param y outputs the predicted response.

   @param ctgWidth is the cardinality of factor-valued response.

   @return void, with output parameter vector.
 */
void Predict::PredictCtg(int y[], int ctgWidth) {
  DecTree::PredictAcrossCtg(y, ctgWidth, 0, 0, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}
