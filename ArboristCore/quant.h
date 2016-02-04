// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file quant.h

   @brief Data structures and methods for predicting and writing quantiles.

   @author Mark Seligman

 */

#ifndef ARBORIST_QUANT_H
#define ARBORIST_QUANT_H

/**
 @brief Quantile signature.
*/
class Quant {
  // Set from Forest and Leaf methods:
  const unsigned int nRow;
  class Forest *forest;
  const int height;
  const int nTree;
  const unsigned int *extent;
  const double *yRanked;
  const unsigned int *rank;
  unsigned int *sCount; // Nonconstant:  hammered by binning mechanism.

  const double *qVec;
  const int qCount;
  const unsigned int qBin;
  
  int *leafPos;
  Quant(unsigned int _nRow, class Forest *forest, double *_yRanked, unsigned int *_rank, unsigned int *_sCount, const double _qVec[], int _qCount, unsigned int _qBin);
  ~Quant();
  void PredictRows(int *predictLeaves, double qPred[]);
  unsigned int SmudgeLeaves(unsigned int &logSmudge);
  void Leaves(const int rowPredict[], double qRow[], unsigned int binSize, unsigned int logSmudge);
  int RanksExact(int leafExtent, int leafOff, int sampRanks[]);
  int RanksSmudge(unsigned int leafExtent, int leafOff, int sampRanks[], unsigned int binSize, unsigned int logSmudge);

 public:
  static void Predict(unsigned int _nRow, class Forest *forest, double *_yRanked, unsigned int *_rank, unsigned int *_sCount, const double _qVec[], int _qCount, unsigned int _qBin, int *predictLeaves, double qPred[]);
};

#endif
