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

#include <vector>

#include "typeparam.h"

typedef pair<double, unsigned int> RankedPair;

/**
 @brief Quantile signature.
*/
class Quant {
  const class PredictReg *predictReg;
  const class LeafReg *leafReg;
  const double *yTrain;
  vector<RankedPair> yRanked;
  const vector<double> &qVec;
  const unsigned int qCount;
  vector<class RankCount> rankCount; // forest-wide, by sample.
  unsigned int logSmudge;
  unsigned int binSize;
  vector<unsigned int> binTemp; // Helper vector.
  vector<unsigned int> sCountSmudge;

  int *leafPos;
  
  unsigned int BinSize(unsigned int nRow, unsigned int qBin, unsigned int &_logSmudge);
  void SmudgeLeaves();
  void Leaves(unsigned int rowBlock, double qRow[]);
  unsigned int RanksExact(unsigned int tIdx, unsigned int leafIdx, vector<unsigned int> &sampRanks);
  unsigned int RanksSmudge(unsigned int tIdx, unsigned int LeafIdx, vector<unsigned int> &sampRanks);

  
 public:
  Quant(const class PredictReg *_predictReg, const class LeafReg *_leafReg, const vector<double> &_qVec, unsigned int qBin);
  void PredictAcross(class PredictReg *predict, unsigned int rowStart, unsigned int rowEnd, double qPred[]);
};

#endif
