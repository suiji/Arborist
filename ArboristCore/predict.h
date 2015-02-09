// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.h

   @brief Interface for front-end entry to prediction methods.

   @author Mark Seligman

 */

#ifndef ARBORIST_PREDICT_H
#define ARBORIST_PREDICT_H

/**
 @brief Interface class for front end.
*/
class Predict {
public:
  static void ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bump[], int _origins[], int _facOff[], int _facSplits[]);
  static void ForestReloadQuant(int nTree, double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]);
  static void PredictOOBQuant(double err[], double quantVec[], int qCount, double qPred[], double predGini[]);
  static void PredictOOBReg(double err[], double predGini[]);
  static void PredictOOBCtg(int conf[], double *error, double predGini[]);
  static void PredictQuant(int nRow, double quantVec[], int qcells, double qPred[], double y[]);
  static void PredictReg(double y[]);
  static void PredictCtg(int y[], int ctgWidth);
};
#endif
