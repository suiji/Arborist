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

#include "framemap.h"
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
vector<double> Predict::Regression(const FramePredict *framePredict,
			 const Forest *forest,
				   const LeafReg *leafReg) {
  //			 vector<double> &yPred) {
  // Non-quantile regression does not employ BagLeaf information.
  vector<double> yPred(framePredict->NRow());
  auto predictReg = make_unique<PredictReg>(framePredict, leafReg, forest->NTree(), yPred);
  predictReg->PredictAcross(forest);
  return yPred;
}


/**
   @brief Static entry for regression case.

   // Only prediction method requiring BagLeaf.
 */
vector<double> Predict::Quantiles(const FramePredict *framePredict,
			const Forest *forest,
			const LeafReg *leafReg,
			const vector<double> &quantVec,
			unsigned int qBin,
				  vector<double> &qPred) {
  vector<double> yPred(framePredict->NRow());
  auto predictReg = make_unique<PredictReg>(framePredict, leafReg, forest->NTree(), yPred);
  auto quant = make_unique<Quant>(predictReg.get(), leafReg, quantVec, qBin);
  predictReg->PredictAcross(forest, quant.get(), &qPred[0]);

  return yPred;
}


/**
   @brief Entry for separate classification prediction.
 */
vector<unsigned int> Predict::Classification(const FramePredict *framePredict,
			     const Forest *forest,
			     const LeafCtg *leafCtg,
			     unsigned int *_census,
			     const vector<unsigned int> &_yTest,
			     unsigned int *_conf,
			     vector<double> &_error,
			     double *_prob) {
  vector<unsigned int> yPred(framePredict->NRow());
  auto predictCtg = make_unique<PredictCtg>(framePredict, leafCtg, forest->NTree(), yPred);
  predictCtg->PredictAcross(forest, _census, _yTest, _conf, _error, _prob);
  return yPred;
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
  predictLeaves = new unsigned int[rowBlock * nTree];
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
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    PredictBlock(forest, rowStart, rowEnd, bag);
    Score(votes, rowStart, rowEnd);
    if (prob != nullptr)
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
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    PredictBlock(forest, rowStart, rowEnd, bag);
    Score(rowStart, rowEnd);
  }
}


/**
   @brief Predictions for a block of rows, with quantiles.

   @return void, with side-effected prediction vectors.
 */
void PredictReg::PredictAcross(const Forest *forest,
			       Quant *quant,
			       double qPred[]) {
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    PredictBlock(forest, rowStart, rowEnd, leafReg->Bag());
    Score(rowStart, rowEnd);
    quant->PredictAcross(this, rowStart, rowEnd, qPred);
  }
}


/**
   @brief Dispatches prediction method based on available predictor types.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Predict::PredictBlock(const Forest *forest,
			   unsigned int rowStart,
			   unsigned int rowEnd,
			   const BitMatrix *bag) {
  if (framePredict->NPredFac() == 0)
    PredictBlockNum(forest, rowStart, rowEnd, bag);
  else if (framePredict->NPredNum() == 0)
    PredictBlockFac(forest, rowStart, rowEnd, bag);
  else
    PredictBlockMixed(forest, rowStart, rowEnd, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Predict::PredictBlockNum(const Forest *forest,
			      unsigned int rowStart,
			      unsigned int rowEnd,
			      const BitMatrix *bag) {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      RowNum(row, row - rowStart, forest->Node(), forest->Origin(), bag);
    }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Predict::PredictBlockFac(const Forest *forest,
			      unsigned int rowStart,
			      unsigned int rowEnd,
			      const BitMatrix *bag) {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      RowFac(row, row - rowStart, forest->Node(), forest->Origin(), forest->FacSplit(), bag);
  }
  }

}


/**
   @brief Multi-row prediction with predictors of both numeric and factor type.

   @param rowStart is the first row in the block.

   @param rowEnd is the first row beyond the block.

   @param bag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Predict::PredictBlockMixed(const Forest *forest,
				unsigned int rowStart,
				unsigned int rowEnd,
				const BitMatrix *bag) {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      RowMixed(row, row - rowStart, forest->Node(), forest->Origin(), forest->FacSplit(), bag);
    }
  }
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



void Predict::RowNum(unsigned int row,
		     unsigned int blockRow,
		     const ForestNode *forestNode,
		     const unsigned int *origin,
		     const class BitMatrix *bag) {
  auto rowT = framePredict->RowNum(blockRow);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      BagIdx(blockRow, tIdx);
      continue;
    }

    auto idx = origin[tIdx];
    auto leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(rowT, leafIdx);
    }

    LeafIdx(blockRow, tIdx, leafIdx);
  }
}


/**
   @brief Prediction with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Predict::RowFac(unsigned int row,
		     unsigned int blockRow,
		     const ForestNode *forestNode,
		     const unsigned int *origin,
		     const BVJagged *facSplit,
		     const class BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      BagIdx(blockRow, tIdx);
      continue;
    }

    auto rowT = framePredict->RowFac(blockRow);
    auto idx = origin[tIdx];
    auto leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(facSplit, rowT, tIdx, leafIdx);
    }

    LeafIdx(blockRow, tIdx, leafIdx);
  }
}

/**
   @brief Prediction with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Predict::RowMixed(unsigned int row,
		       unsigned int blockRow,
		       const ForestNode *forestNode,
		       const unsigned int *origin,
		       const BVJagged *facSplit,
		       const BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      BagIdx(blockRow, tIdx);
      continue;
    }

    auto rowNT = framePredict->RowNum(blockRow);
    auto rowFT = framePredict->RowFac(blockRow);
    auto idx = origin[tIdx];
    auto leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(framePredict, facSplit, rowFT, rowNT, tIdx, leafIdx);
    }

    LeafIdx(blockRow, tIdx, leafIdx);
  }
}
