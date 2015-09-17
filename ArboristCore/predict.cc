// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.cc

   @brief Methods for prediction.

   @author Mark Seligman
 */

#include "forest.h"
#include "predict.h"
#include "quant.h"

int Predict::nTree = 0;
unsigned int Predict::nRow = 0;
unsigned int Predict::ctgWidth = 0;
ForestReg *Predict::forestReg = 0;
ForestCtg *Predict::forestCtg = 0;

/**
   @brief Entry for separate prediction.
 */
void Predict::ForestCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], unsigned int _ctgWidth, double *_leafWeight) {
  unsigned _nRow = Forest::PredImmutables();
  Immutables(_nTree, _nRow, _ctgWidth);
  forestCtg = Forest::FactoryCtg(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit, _ctgWidth, _leafWeight);
}


/**
   @brief Entry for separate prediction.

   @return void.
 */
void Predict::ForestReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], int _rank[], int _sCount[], double _yRanked[]) {
  unsigned _nRow = Forest::PredImmutables();
  Immutables(_nTree, _nRow);
  forestReg = Forest::FactoryReg(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit, _rank, _sCount, _yRanked);
}


void Predict::Immutables(int _nTree, int _nRow, int _ctgWidth) {
  nTree = _nTree;
  nRow = _nRow;
  ctgWidth = _ctgWidth;
}


void Predict::DeImmutables() {
  nTree = -1;
  ctgWidth = 0;
  forestReg = 0;
  forestCtg = 0;
}

/**
   @brief Static entry for validation.
 */
int Predict::ValidateCtg(const int yCtg[], const unsigned int *bag, int yPred[], int *census, int *conf, double error[], double *prob) {
  if (!forestCtg)
    return -1;

  int *predictLeaves = new int[nRow * nTree];
  forestCtg->Predict(predictLeaves, bag);

  double *votes = forestCtg->Score(predictLeaves);
  Vote(votes, census, yPred);
  delete [] votes;
  Validate(yCtg, yPred, conf, error);

  if (prob != 0)
    forestCtg->Prob(predictLeaves, prob);
  delete [] predictLeaves;
  Forest::DeFactory(forestCtg);
  
  return 0;
}


/**
   @brief Static entry for prediction.

   @return 0 for normal termination, -1 for exception.
 */
int Predict::PredictCtg(int yPred[], int *census, double *prob) {
  if (!forestCtg)
    return -1;

  int *predictLeaves = new int[nRow * nTree];
  forestCtg->Predict(predictLeaves, 0);

  double *votes = forestCtg->Score(predictLeaves);
  Vote(votes, census, yPred);
  delete [] votes;

  if (prob != 0) {
    forestCtg->Prob(predictLeaves, prob);
  }
  delete [] predictLeaves;

  Forest::DeFactory(forestCtg);

  return 0;
}


/**
   @brief Fills in confusion matrix and error vector.

   @param yCtg contains the training response.

   @param yPred is the predicted response.

   @param confusion is the confusion matrix.

   @param error outputs the classification errors.

   @return void.
*/
void Predict::Validate(const int yCtg[], const int yPred[], int confusion[], double error[]) {
  for (unsigned int row = 0; row < nRow; row++) {
    confusion[ctgWidth * yCtg[row] + yPred[row]]++;
  }

  // Fills in classification error vector from off-diagonal confusion elements..
  //
  for (unsigned int rsp = 0; rsp < ctgWidth; rsp++) {
    int numWrong = 0;
    int numRight = 0;
    for (unsigned int predicted = 0; predicted < ctgWidth; predicted++) {
      if (predicted != rsp) {  // Mispredictions are off-diagonal.
        numWrong += confusion[ctgWidth * rsp + predicted];
      }
      else {
	numRight = confusion[ctgWidth * rsp + predicted];
      }
    }
    error[rsp] = double(numWrong) / double(numWrong + numRight);
  }
}

 
/**
   @brief Voting for non-bagged prediction.  Rounds jittered scores to category.

   @param predictLeaves are the predicted terminal indices.

   @param yCtg outputs predicted response.

   @return void, with output reference vector.
*/
void Predict::Vote(double *votes, int census[], int yPred[]) {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    int argMax = -1;
    double scoreMax = 0.0;
    double *score = votes + row * ctgWidth;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      double ctgScore = score[ctg]; // Jittered vote count.
      if (ctgScore > scoreMax) {
	scoreMax = ctgScore;
	argMax = ctg;
      }
      census[row * ctgWidth + ctg] = ctgScore; // De-jittered.
    }
    yPred[row] = argMax;
  }
  }
}

 
/**
   @brief Static entry with local leaf allocation.
 */
int Predict::PredictReg(double yPred[], const unsigned int *bag) {
  if (!forestReg) {
    return -1;
  }

  int *predictLeaves = new int[nRow * nTree];
  forestReg->Predict(yPred, predictLeaves, bag);
  delete [] predictLeaves;

  Forest::DeFactory(forestReg);
  return 0;
}


/**
 */
int Predict::PredictQuant(double yPred[], const double qVec[], int qCount, unsigned int qBin, double qPred[], const unsigned int *bag) {
  if (!forestReg) {
    return -1;
  }
  int *predictLeaves = new int[nRow * nTree];
  forestReg->Predict(yPred, predictLeaves, bag);
  Quant::Predict(forestReg, qVec, qCount, qBin, predictLeaves, qPred);

  delete [] predictLeaves;
  return 0;
}
