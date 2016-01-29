// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predblock.cc

   @brief Methods for blocks of similarly-typed predictors.

   @author Mark Seligman
 */

#include "predblock.h"

unsigned int PredBlock::nPredNum = 0;
unsigned int PredBlock::nPredFac = 0;
unsigned int PredBlock::nRow = 0;
double *PBTrain::feNum = 0;
double *PBPredict::feNumT = 0;
int *PBPredict::feFacT = 0;
int *PBTrain::feCard = 0; // Factor predictor cardinalities.
unsigned int PBTrain::cardMax = 0;  // High watermark of factor cardinalities.


/**
   @brief Static initialization for training.

   @return void.
 */
void PBTrain::Immutables(double *_feNum, int *_feCard, const int _cardMax, const unsigned int _nPredNum, const unsigned int _nPredFac, const unsigned int _nRow) {
  feNum = _feNum;
  feCard = _feCard;
  cardMax = _cardMax;
  nPredNum = _nPredNum;
  nPredFac = _nPredFac;
  nRow = _nRow;
}


/**
   @brief Static initialization for prediction.

   @return void.
 */
void PBPredict::Immutables(double *_feNumT, int *_feFacT, const unsigned int _nPredNum, const unsigned int _nPredFac, const unsigned int _nRow) {
  feNumT = _feNumT;
  feFacT = _feFacT;
  nPredNum = _nPredNum;
  nPredFac = _nPredFac;
  nRow = _nRow;
}


void PredBlock::DeImmutables() {
  nPredNum = nPredFac = nRow = 0;
}

/**
   @breif De-initializes all statics.

   @return void.
 */
void PBTrain::DeImmutables() {
  feNum = 0;
  feCard = 0; // Factor predictor cardinalities.
  nPredNum = nPredFac =  nRow = 0;
  cardMax = 0;  // High watermark of factor cardinalities.
  PredBlock::DeImmutables();
}


void PBPredict::DeImmutables() {
  feNumT = 0;
  feFacT = 0;
  PredBlock::DeImmutables();
}

/**
   @brief Estimates mean of a numeric predictor from values at two rows.
   N.B.:  assumes 'predIdx' and 'feIdx' are identical for numeric
   predictors; otherwise remap with predMap[].

   @param predIdx is the core-ordered predictor index.

   @param rowLow is the row index of the lower estimate.

   @param rowHigh is the row index of the higher estimate.

   @return arithmetic mean of the (possibly equal) estimates.
 */
double PBTrain::MeanVal(int predIdx, int rowLow, int rowHigh) {
  double *feBase = &feNum[predIdx * nRow];
  return 0.5 * (feBase[rowLow] + feBase[rowHigh]);
}
