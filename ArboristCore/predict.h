// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.h

   @brief Data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef ARBORIST_PREDICT_H
#define ARBORIST_PREDICT_H


class Predict {
  static int nTree;
  static unsigned int nRow;
  static unsigned int ctgWidth;

  static void Immutables(int _nTree, int _nRow, int _ctgWidth = 0);
  static void DeImmutables();

  static class ForestReg *forestReg;
  static class ForestCtg *forestCtg;

  static void Validate(const int yCtg[], const int yPred[], int confusion[], double error[]);
  static void Vote(double *score, int *census, int yCtg[]);
 public:
  static void ForestCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], unsigned int _ctgWidth, double *_leafWeight);
  static void ForestReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], int _rank[], int _sCount[], double *_yRanked);
  static int ValidateCtg(const int yCtg[], const unsigned int *bag, int yPred[], int *census, int *conf, double error[], double *prob);
  static int PredictCtg(int yPred[], int *census, double *prob);
  static int PredictReg(double yPred[], const unsigned int *bag);
  static int PredictQuant(double yPred[], const double qVec[], int qCount, unsigned int qBin, double qPred[], const unsigned int *bag);
};

#endif
