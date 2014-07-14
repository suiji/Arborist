/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// Interface class for front end.
// Holds simulation-specific parameters of the data.
//

#include "predict.h"
#include "predictor.h"
#include "dectree.h"
#include "response.h"


int QuantSig::qCells = 0;
double *QuantSig::qVec = 0;
double *QuantSig::qPred = 0;

void QuantSig::Factory(double *_qVec, int _qCells, double *_qPred) {
  qCells = _qCells;
  qVec = _qVec;
  qPred = _qPred;
}
void QuantSig::DeFactory() {
  qCells = 0;
  qVec = qPred = 0;
}

// Thin-layer interfaces.
//
void Predict::ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bumpL[], int _bumpR[], int _origins[], int _facOff[], int _facSplits[]) {
  DecTree::ForestReload(_nTree, _forestSize, _preds, _splits, _scores, _bumpL, _bumpR, _origins, _facOff, _facSplits);
}

void Predict::ForestReloadQuant(double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]) {
  DecTree::ForestReloadQuant(qYRanked, qYLen, qRankOrigin, qRank, qRankCount, qLeafPos, qLeafExtent);
}

void Predict::Finish(double *predGini) {
  DecTree::ScaleGini(predGini);
  Response::DeFactory();
}

void Predict::PredictOOBQuant(double *err, double *quantVec, int qCells, double *qPred, double *predGini) {
  QuantSig::Factory(quantVec, qCells, qPred);
  Response::response->PredictOOB(0, err);
  QuantSig::DeFactory();
  Finish(predGini);
}

void Predict::PredictOOBReg(double *err, double predGini[]) {
  Response::response->PredictOOB(0, err);
  Finish(predGini);
}

void Predict::PredictOOBCtg(int conf[], double *error, double predGini[]) {
  Response::response->PredictOOB(conf, error);
  Finish(predGini);
}

void Predict::PredictQuant(double quantVec[], const int qCells, double qPred[], double y[]) {
  QuantSig::Factory(quantVec, qCells, qPred);
  DecTree::PredictAcrossReg(y, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}

void Predict::PredictReg(double y[]) {
  DecTree::PredictAcrossReg(y, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}

void Predict::PredictCtg(int y[], int ctgWidth) {
  DecTree::PredictAcrossCtg(y, ctgWidth, 0, 0, false);
  DecTree::DeForestPredict();
  Predictor::DeFactory();
}
