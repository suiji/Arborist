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


/**
 @brief Quantile signature.
*/
class Quant {
  const class Forest *forest;
  const int height;
  const int nTree;
  const std::vector<double> &yRanked;
  const std::vector<unsigned int> &rank;
  const std::vector<unsigned int> &sCount;
  const std::vector<double> &qVec;
  const int qCount;
  unsigned int logSmudge;
  unsigned int binSize;
  unsigned int *sCountSmudge;

  int *leafPos;
  
  unsigned int BinSize(unsigned int nRow, unsigned int qBin, unsigned int &_logSmudge);
  void SmudgeLeaves();
  void Leaves(const int rowLeaves[], double qRow[]);
  int RanksExact(int leafExtent, int leafOff, int sampRanks[]);
  int RanksSmudge(unsigned int leafExtent, int leafOff, int sampRanks[]);
 public:
  Quant(const class Forest *_forest, const std::vector<double> &_yRanked, const std::vector<unsigned int> &_rank, const std::vector<unsigned int> &_sCount, const std::vector<double> &_qVec, unsigned int qBin);
  ~Quant();
  void PredictAcross(const int *predictLeaves, unsigned int rowStart, unsigned int rowEnd, double qPred[]);
};

#endif
