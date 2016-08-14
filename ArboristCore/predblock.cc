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
double *PBPredict::feNumT = 0;
int *PBPredict::feFacT = 0;

const double *PBTrain::feNum = 0;
const unsigned int *PBTrain::feCard = 0;
unsigned int PBTrain::cardMax = 0;  // High watermark of factor cardinalities.


/**
   @brief Static initialization for training.

   @return void.
 */
void PBTrain::Immutables(const double _feNum[], const unsigned int _feCard[], unsigned int _cardMax, unsigned int _nPredNum, unsigned int _nPredFac, unsigned int _nRow) {
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
void PBPredict::Immutables(double *_feNumT, int *_feFacT, unsigned int _nPredNum, unsigned int _nPredFac, unsigned int _nRow) {
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
