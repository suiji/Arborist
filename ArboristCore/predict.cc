// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.cc

   @brief Methods for validation and prediction.

   @author Mark Seligman
 */

#include "predblock.h"
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "quant.h"
#include "bv.h"

//#include <iostream>
using namespace std;


/**
   @brief Static entry for regression case.
 */
void Predict::Regression(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<unsigned int> &_rank, std::vector<double> &yPred, unsigned int bagTrain) {
  int nTree = _origin.size();
  unsigned int _nRow = yPred.size();
  PBPredict::Immutables(_blockNumT, _blockFacT, _nPredNum, _nPredFac, _nRow);
  LeafReg *leafReg = new LeafReg(_leafOrigin, _leafNode, _bagRow, _rank);
  PredictReg *predictReg = new PredictReg(leafReg, nTree, _nRow, _leafNode.size());
  Forest *forest =  new Forest(_forestNode, _origin, _facOff, _facSplit, predictReg);
  BitMatrix *bag = leafReg->ForestBag(bagTrain);
  predictReg->PredictAcross(forest, yPred, bag);

  delete bag;
  delete predictReg;
  delete forest;
  delete leafReg;
  PBPredict::DeImmutables();
}


/**
   @brief Static entry for regression case.
 */
void Predict::Quantiles(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<unsigned int> &_rank, const std::vector<double> &yRanked, std::vector<double> &yPred, const std::vector<double> &quantVec, unsigned int qBin, std::vector<double> &qPred, unsigned int bagTrain) {
  int nTree = _origin.size();
  unsigned int _nRow = yPred.size();
  PBPredict::Immutables(_blockNumT, _blockFacT, _nPredNum, _nPredFac, _nRow);
  LeafReg *leafReg = new LeafReg(_leafOrigin, _leafNode, _bagRow, _rank);
  PredictReg *predictReg = new PredictReg(leafReg, nTree, _nRow, _leafNode.size());
  Forest *forest =  new Forest(_forestNode, _origin, _facOff, _facSplit, predictReg);
  BitMatrix *bag = leafReg->ForestBag(bagTrain);
  Quant *quant = new Quant(predictReg, leafReg, yRanked, quantVec, qBin);
  predictReg->PredictAcross(forest, yPred, quant, &qPred[0], bag);

  delete bag;
  delete predictReg;
  delete leafReg;
  delete quant;
  delete forest;
  
  PBPredict::DeImmutables();
}


/**
   @brief Entry for separate classification prediction.
 */
void Predict::Classification(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_leafInfoCtg, std::vector<int> &yPred, int *_census, const std::vector<unsigned int> &_yTest, int *_conf, std::vector<double> &_error, double *_prob, unsigned int bagTrain) {
  int nTree = _origin.size();
  unsigned int _nRow = yPred.size();
  PBPredict::Immutables(_blockNumT, _blockFacT, _nPredNum, _nPredFac, _nRow);
  LeafCtg *leafCtg = new LeafCtg(_leafOrigin, _leafNode, _bagRow, _leafInfoCtg);
  PredictCtg *predictCtg = new PredictCtg(leafCtg, nTree, _nRow, _leafNode.size());
  Forest *forest = new Forest(_forestNode, _origin, _facOff, _facSplit, predictCtg);
  BitMatrix *bag = leafCtg->ForestBag(bagTrain);
  predictCtg->PredictAcross(forest, bag, _census, yPred, _yTest, _conf, _error, _prob);

  delete predictCtg;
  delete forest;
  delete leafCtg;
  delete bag;
  PBPredict::DeImmutables();
}


PredictCtg::PredictCtg(const LeafCtg *_leafCtg, int _nTree, unsigned int _nRow, unsigned int _nonLeafIdx) : Predict(_nTree, _nRow, _nonLeafIdx), leafCtg(_leafCtg), ctgWidth(leafCtg->CtgWidth()) {
}


PredictReg::PredictReg(const LeafReg *_leafReg, int _nTree, unsigned int _nRow, unsigned int _nonLeafIdx) : Predict(_nTree, _nRow, _nonLeafIdx), leafReg(_leafReg) {
}


Predict::Predict(int _nTree, unsigned int _nRow, unsigned int _nonLeafIdx) : nonLeafIdx(_nonLeafIdx), nTree(_nTree), nRow(_nRow) {
  predictLeaves = new unsigned int[rowBlock * nTree];
}


Predict::~Predict() {
  delete [] predictLeaves;
}


void PredictCtg::PredictAcross(const Forest *forest, const BitMatrix *bag, int *census, std::vector<int> &yPred, const std::vector<unsigned int> &yTest, int *conf, std::vector<double> &error, double *prob) {
  double *votes = new double[nRow * ctgWidth];
  for (unsigned int i = 0; i < nRow * ctgWidth; i++)
    votes[i] = 0;
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    forest->PredictAcross(rowStart, rowEnd, bag);
    Score(votes, rowStart, rowEnd);
    if (prob != 0)
      Prob(prob, rowStart, rowEnd);
  }
  Vote(votes, census, &yPred[0]);
  delete [] votes;

  if (yTest.size() > 0) {
    Validate(yTest, &yPred[0], conf, error);
  }
}


/**
   @brief Fills in confusion matrix and error vector.

   @param yTest contains the test response.

   @param yPred is the predicted response.

   @param confusion is the confusion matrix.

   @param error outputs the classification errors.

   @return void.
*/
void PredictCtg::Validate(const std::vector<unsigned int> &yTest, const int yPred[], int confusion[], std::vector<double> &error) {
  for (unsigned int row = 0; row < nRow; row++) {
    confusion[ctgWidth * yTest[row] + yPred[row]]++;
  }

  // Fills in classification error vector from off-diagonal confusion elements..
  //
  for (unsigned int rsp = 0; rsp < error.size(); rsp++) {
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
    error[rsp] = numWrong + numRight == 0 ? 0.0 : double(numWrong) / double(numWrong + numRight);
  }
}

 
/**
   @brief Voting for non-bagged prediction.  Rounds jittered scores to category.

   @param yCtg outputs predicted response.

   @return void, with output reference vector.
*/
void PredictCtg::Vote(double *votes, int census[], int yPred[]) {
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
   @brief Computes score from leaf predictions.

   @return internal vote table, with output reference vector.
 */
void PredictCtg::Score(double *votes, unsigned int rowStart, unsigned int rowEnd) {
  unsigned int blockRow;

// TODO:  Recast loop by blocks, to avoid
// false sharing.
#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < rowEnd - rowStart; blockRow++) {
    double *prediction = votes + (rowStart + blockRow) * ctgWidth;
    for (int tc = 0; tc < nTree; tc++) {
      if (!IsBagged(blockRow, tc)) {
	double val = leafCtg->GetScore(tc, LeafIdx(blockRow, tc));
	unsigned int ctg = val; // Truncates jittered score for indexing.
	prediction[ctg] += 1 + val - ctg;
      }
    }
  }
  }
}


void PredictCtg::Prob(double *prob, unsigned int rowStart, unsigned int rowEnd) {
  for (unsigned int blockRow = 0; blockRow < rowEnd - rowStart; blockRow++) {
    double *probRow = prob + (rowStart + blockRow) * ctgWidth;
    double rowSum = 0.0;
    for (int tc = 0; tc < nTree; tc++) {
      if (!IsBagged(blockRow, tc)) {
	for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	  double idxWeight = leafCtg->WeightCtg(tc, LeafIdx(blockRow, tc), ctg);
	  probRow[ctg] += idxWeight;
	  rowSum += idxWeight;
	}
      }
    }
    double recipSum = 1.0 / rowSum;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++)
      probRow[ctg] *= recipSum;
  }
}


/**
 */
void PredictReg::PredictAcross(const Forest *forest, std::vector<double> &yPred, const BitMatrix *bag) {
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    forest->PredictAcross(rowStart, rowEnd, bag);
    Score(rowStart, rowEnd, &yPred[rowStart]);
  }
}


/**
   @brief Predictions for a block of rows, with quantiles.

   @return void, with side-effected prediction vectors.
 */
void PredictReg::PredictAcross(const Forest *forest, std::vector<double> &yPred, Quant *quant, double qPred[], const BitMatrix *bag) {
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    forest->PredictAcross(rowStart, rowEnd, bag);
    Score(rowStart, rowEnd, &yPred[rowStart]);
    quant->PredictAcross(rowStart, rowEnd, qPred);
  }
}



/**
  @brief Sets regression scores from leaf predictions.

  @param yPred outputs the score predictions.

  @return void, with output refererence vector.
 */
void PredictReg::Score(unsigned int rowStart, unsigned int rowEnd, double yPred[]) {
  unsigned int blockRow;

#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
    for (blockRow = 0; blockRow < rowEnd - rowStart; blockRow++) {
      double score = 0.0;
      int treesSeen = 0;
      for (int tc = 0; tc < nTree; tc++) {
        if (!IsBagged(blockRow, tc)) {
          treesSeen++;
          score += leafReg->GetScore(tc, LeafIdx(blockRow, tc));
        }
      }
      // Will be NA if row not oob at least once:
      yPred[blockRow] = score / treesSeen;
    }
  }
}
