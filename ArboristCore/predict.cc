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

#include "frameblock.h"
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "quant.h"
#include "bv.h"

#include <cfloat>
#include <algorithm>


/**
   @brief Static entry for regression case.
 */
void Predict::Regression(const FramePredict *framePredict,
			 const Forest *forest,
			 const LeafReg *leafReg,
			 vector<double> &_yPred) {
  // Non-quantile regression does not employ BagLeaf information.
  PredictReg *predictReg = new PredictReg(framePredict, leafReg, forest->NTree(), _yPred);
  predictReg->PredictAcross(forest);

  delete predictReg;
}


/**
   @brief Static entry for regression case.

   // Only prediction method requiring BagLeaf.
 */
void Predict::Quantiles(const FramePredict *framePredict,
			const Forest *forest,
			const LeafReg *leafReg,
			vector<double> &_yPred,
			const vector<double> &quantVec,
			unsigned int qBin,
			vector<double> &qPred,
			bool validate) {
  PredictReg *predictReg = new PredictReg(framePredict, leafReg, forest->NTree(), _yPred);
  Quant *quant = new Quant(predictReg, leafReg, quantVec, qBin);
  predictReg->PredictAcross(forest, quant, &qPred[0], validate);

  delete predictReg;
  delete quant;
}


/**
   @brief Entry for separate classification prediction.
 */
void Predict::Classification(const FramePredict *framePredict,
			     const Forest *forest,
			     const LeafCtg *leafCtg,
			     vector<unsigned int> &_yPred,
			     unsigned int *_census,
			     const vector<unsigned int> &_yTest,
			     unsigned int *_conf,
			     vector<double> &_error,
			     double *_prob) {
  PredictCtg *predictCtg = new PredictCtg(framePredict, leafCtg, forest->NTree(), _yPred);
  predictCtg->PredictAcross(forest, _census, _yTest, _conf, _error, _prob);

  delete predictCtg;
}


PredictCtg::PredictCtg(const FramePredict *_framePredict,
		       const LeafCtg *_leafCtg,
		       unsigned int _nTree,
		       vector<unsigned int> &_yPred) :
  Predict(_framePredict, _nTree, _yPred.size(), _leafCtg->NoLeaf()), leafCtg(_leafCtg), ctgWidth(leafCtg->CtgWidth()), yPred(_yPred), defaultScore(ctgWidth), defaultWeight(vector<double>(ctgWidth)) {
  fill(defaultWeight.begin(), defaultWeight.end(), -1.0);
}


PredictReg::PredictReg(const FramePredict *_framePredict,
		       const LeafReg *_leafReg,
		       unsigned int _nTree,
		       vector<double> &_yPred) :
  Predict(_framePredict, _nTree, _yPred.size(), _leafReg->NoLeaf()), leafReg(_leafReg), yPred(_yPred), defaultScore(leafReg->MeanTrain()) {
}


/**
   @brief Lazily sets default score.

   @return default score.
 */
unsigned int PredictCtg::DefaultScore() {
  if (defaultScore >= ctgWidth) {
    DefaultInit();

    defaultScore = 0;
    double weightMax = defaultWeight[0];
    for (unsigned int ctg = 1; ctg < ctgWidth; ctg++) {
      if (defaultWeight[ctg] > weightMax) {
	defaultScore = ctg;
	weightMax = defaultWeight[ctg];
      }
    }
  }

  return defaultScore;
}


/**
   @brief Lazily sets default weight.
   TODO:  Ensure error if called when no bag present.

   @return void.
 */
void PredictCtg::DefaultInit() {
  if (defaultWeight[0] < 0.0) { // Unseen.
    fill(defaultWeight.begin(), defaultWeight.end(), 0.0);
    leafCtg->DefaultWeight(defaultWeight);
  }
}


double PredictCtg::DefaultWeight(double *weightPredict) {
  double rowSum = 0.0;
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    weightPredict[ctg] = defaultWeight[ctg];
    rowSum += weightPredict[ctg];
  }

  return rowSum;
}


Predict::Predict(const FramePredict *_framePredict,
		 unsigned int _nTree,
		 unsigned int _nRow,
		 unsigned int _noLeaf) :
  noLeaf(_noLeaf),
  framePredict(_framePredict),
  nTree(_nTree),
  nRow(_nRow) {
  predictLeaves = new unsigned int[FramePredict::rowBlock * nTree];
}


Predict::~Predict() {
  delete [] predictLeaves;
}


PredictCtg::~PredictCtg() {
}


void PredictCtg::PredictAcross(const Forest *forest,
			       unsigned int *census,
			       const vector<unsigned int> &yTest,
			       unsigned int *conf,
			       vector<double> &error,
			       double *prob) {
  const BitMatrix *bag = leafCtg->Bag();

  double *votes = new double[nRow * ctgWidth];
  for (unsigned int i = 0; i < nRow * ctgWidth; i++)
    votes[i] = 0;
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += FramePredict::rowBlock) {
    unsigned int rowEnd = min(rowStart + FramePredict::rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    forest->PredictAcross(this, rowStart, rowEnd, bag);
    Score(votes, rowStart, rowEnd);
    if (prob != 0)
      Prob(prob, rowStart, rowEnd);
  }

  Vote(votes, census);
  delete [] votes;

  if (yTest.size() > 0) {
    Validate(yTest, conf, error);
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
void PredictCtg::Validate(const vector<unsigned int> &yTest,
			  unsigned int confusion[],
			  vector<double> &error) {
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
void PredictCtg::Vote(double *votes,
		      unsigned int census[]) {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < int(nRow); row++) {
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
void PredictCtg::Score(double *votes,
		       unsigned int rowStart,
		       unsigned int rowEnd) {
  int blockRow;

// TODO:  Recast loop by blocks, to avoid
// false sharing.
#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < int(rowEnd - rowStart); blockRow++) {
    double *prediction = votes + (rowStart + blockRow) * ctgWidth;
    unsigned int treesSeen = 0;
    for (unsigned int tc = 0; tc < nTree; tc++) {
      if (!IsBagged(blockRow, tc)) {
	treesSeen++;
	double val = leafCtg->GetScore(tc, LeafIdx(blockRow, tc));
	unsigned int ctg = val; // Truncates jittered score for indexing.
	prediction[ctg] += 1 + val - ctg;
      }
    }
    if (treesSeen == 0) {
      for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	prediction[ctg] = 0.0;
      }
      prediction[DefaultScore()] = 1;
    }
  }
  }
}


void PredictCtg::Prob(double *prob,
		      unsigned int rowStart,
		      unsigned int rowEnd) {
  for (unsigned int blockRow = 0; blockRow < rowEnd - rowStart; blockRow++) {
    double *probRow = prob + (rowStart + blockRow) * ctgWidth;
    double rowSum = 0.0;
    unsigned int treesSeen = 0;
    for (unsigned int tc = 0; tc < nTree; tc++) {
      if (!IsBagged(blockRow, tc)) {
	treesSeen++;
	for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	  double idxWeight = leafCtg->WeightCtg(tc, LeafIdx(blockRow, tc), ctg);
	  probRow[ctg] += idxWeight;
	  rowSum += idxWeight;
	}
      }
    }
    if (treesSeen == 0) {
      rowSum = DefaultWeight(probRow);
    }

    double scale = 1.0 / rowSum;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++)
      probRow[ctg] *= scale;
  }
}


/**
 */
void PredictReg::PredictAcross(const Forest *forest) {
  const BitMatrix *bag = leafReg->Bag();
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += FramePredict::rowBlock) {
    unsigned int rowEnd = min(rowStart + FramePredict::rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    forest->PredictAcross(this, rowStart, rowEnd, bag);
    Score(rowStart, rowEnd);
  }
}


/**
   @brief Predictions for a block of rows, with quantiles.

   @return void, with side-effected prediction vectors.
 */
void PredictReg::PredictAcross(const Forest *forest,
			       Quant *quant,
			       double qPred[],
			       bool validate) {
  const BitMatrix *leafBag = validate ? leafReg->Bag() : new BitMatrix(0, 0);
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += FramePredict::rowBlock) {
    unsigned int rowEnd = min(rowStart + FramePredict::rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    forest->PredictAcross(this, rowStart, rowEnd, leafBag);
    Score(rowStart, rowEnd);
    quant->PredictAcross(this, rowStart, rowEnd, qPred);
  }

  if (!validate)
    delete leafBag;
}



/**
  @brief Sets regression scores from leaf predictions.

  @return void, with output refererence vector.
 */
void PredictReg::Score(unsigned int rowStart,
		       unsigned int rowEnd) {
  int blockRow;

#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < int(rowEnd - rowStart); blockRow++) {
      double score = 0.0;
      int treesSeen = 0;
      for (unsigned int tc = 0; tc < nTree; tc++) {
        if (!IsBagged(blockRow, tc)) {
          treesSeen++;
          score += leafReg->GetScore(tc, LeafIdx(blockRow, tc));
        }
      }
      yPred[rowStart + blockRow] = treesSeen > 0 ? score / treesSeen : defaultScore;
    }
  }
}


const double *Predict::RowNum(unsigned rowOff) const {
  return framePredict->RowNum(rowOff);
}



const unsigned int *Predict::RowFac(unsigned int rowOff) const {
  return framePredict->RowFac(rowOff);
}
