// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman

   These methods are mostly mechanical.  Several methods are tasked
   with populating or depopulating tree-related data structures.  The
   tree-walking methods are clones of one another, with slight variations
   based on response or predictor type.
 */


#include "predictor.h"
#include "forest.h"

//#include <iostream>
using namespace std;

int Forest::nPred = -1; // et seq.:  observation-derived immutables.
int Forest::nPredNum = -1;
int Forest::nPredFac = -1;
unsigned int Forest::nRow = 0;

ForestReg *Forest::FactoryReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], int _rank[], int _sCount[], double _yRanked[]) {
  return new ForestReg(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit, _rank, _sCount, _yRanked);
}


ForestCtg *Forest::FactoryCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], unsigned int _ctgWidth, double _leafWeight[]) {
  return new ForestCtg(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit, _ctgWidth, _leafWeight);
}


/**
   @brief Sets per-session immutables describing predictor layout.
   TODO:  Replace with predictor-based factory.

   @return row count, temporarily:  TODO repair.
 */
unsigned int Forest::PredImmutables() {
  nRow = Predictor::NRow();
  nPred = Predictor::NPred();
  nPredNum = Predictor::NPredNum();
  nPredFac = Predictor::NPredFac();

  return nRow;
}


/**
   @brief Unsets per-session static values.
 */
void Forest::PredDeImmutables() {
  nRow = 0;
  nPred = nPredNum = nPredFac = -1;
}


/**
   @brief Deletes forest object and tells Predictor to shut down.

   @parm forest is the forest object.

   @return void.
 */
void Forest::DeFactory(Forest *forest) {
  delete forest;

  PredDeImmutables();
  Predictor::DeFactory();
}


/**
   @brief Reloading classification constructor, using front-end storage.

   @param _nTree is the number of trees in the forest.

   @param _forestSize is the length of the multi-vector holding all tree parameters.

   @param _preds[] are the predictors associated with tree nonterminals.

   @param _splits[] are the splitting values associated with nonterminals, or scores.

   @param _bump[] are the increments from node to LH successor.

   @param _origins[] are the offsets into the multivector denoting each individual tree vector.

   @param _facOff[] are the offsets into the multi-bitvector denoting each tree's factor splitting values.

   @param _facSplits[] are the factor splitting values. 

   @param _leafWeight holds the class-weighted sample values for each leaf.
*/
ForestCtg::ForestCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], unsigned int _ctgWidth, double *_leafWeight) : Forest(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit), ctgWidth(_ctgWidth), leafWeight(_leafWeight) {
}


/**
   @brief Reloading regression constructor.  As above, but with the following:

   @param _rank[] are the sample ranks.

   @param _sCount[] are the sample multiplicities.
 */
ForestReg::ForestReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], int _rank[], int _sCount[], double _yRanked[]) : Forest(_nTree, _forestSize, _preds, _splits, _bump, _origins, _facOff, _facSplit), rank(_rank), sCount(_sCount) , yRanked(_yRanked) {
}


/**
   @brief Base class reload constructor.  Parameters common to Ctg, Reg variants.
*/
Forest::Forest(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[]) : nTree(_nTree), treeOrigin(_origins), facOff(_facOff), facSplit(_facSplit), pred(_preds), num(_splits), bump(_bump), forestSize(_forestSize) {
}


/**
 */
void ForestCtg::Predict(int *predictLeaves, const unsigned int bag[]) {
  PredictAcross(predictLeaves, bag);
}


/**
   @brief Computes score from leaf predictions.

   @param predictLeaves are the predicted leaf indices.

   @return internal vote table, with output reference vector.
 */
double *ForestCtg::Score(int *predictLeaves) {
  unsigned int row;
  double *votes = new double[nRow * ctgWidth];
  for (row = 0; row < nRow * ctgWidth; row++)
    votes[row] = 0.0;
  
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    int *leaves = predictLeaves + row * nTree;
    double *prediction = votes + row * ctgWidth;
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leaves[tc];
      if (leafIdx >= 0) {
	double val = *(num + treeOrigin[tc] + leafIdx);
	unsigned int ctg = val; // Truncates jittered score for indexing.
	prediction[ctg] += 1 + val - ctg;
      }
    }
  }
  }

  return votes;
}
  

/**
   @brief Fills in the probability matrix from leaf scores.

   @param predictLeaves are the predicted leaf indices.

   @param prob outputs the leaf scores.

   @return void.
 */ 
void ForestCtg::Prob(int *predictLeaves, double *prob) {
  for (unsigned int row = 0; row < nRow; row++) {
    int *leafRow = predictLeaves + row * nTree;
    double *probRow = prob + row * ctgWidth;
    double rowSum = 0.0;
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leafRow[tc];
      if (leafIdx >= 0) {
        double *idxWeight = leafWeight + ctgWidth * (treeOrigin[tc] + leafIdx);
	for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	  probRow[ctg] += idxWeight[ctg];
	  rowSum += idxWeight[ctg];
	}
      }
    }
    double recipSum = 1.0 / rowSum;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++)
      probRow[ctg] *= recipSum;
  }
}


/**
   @brief Regression prediction.

   @return void.
 */
void ForestReg::Predict(double yPred[], int predictLeaves[], const unsigned int bag[]) {
  PredictAcross(predictLeaves, bag);
  Score(predictLeaves, yPred);
}


/**
  @brief Sets regression scores from leaf predictions.

  @param predictLeaves holds the leaf predictions.

  @param yPred outputs the score predictions.

  @return void, with output refererence vector.
 */
void ForestReg::Score(int predictLeaves[], double yPred[]) {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    int *leaves = predictLeaves + row * nTree;
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leaves[tc];
      if (leafIdx >= 0) {
        treesSeen++;
        score +=  num[treeOrigin[tc] + leafIdx];
      }
    }
    yPred[row] = score / treesSeen; // Assumes >= 1 tree seen.
  }
  }
}

/**
   @brief Call-back enabling Quant class to access forest fields.

   @return Height of forest.
 */
int ForestReg::QuantFields(int &_nTree, unsigned int &_nRow, int *&_origin, int *&_nonTerm, int *&_extent, double *&_yRanked, int *&_rank, int *&_sCount) const {
  _nTree = nTree;
  _nRow = nRow;
  _origin = treeOrigin;
  _nonTerm = &bump[0];
  _extent = &pred[0];
  _yRanked = yRanked;
  _rank = &rank[0];
  _sCount = &sCount[0];

  return forestSize;
}


/**
 */
void Forest::PredictAcross(int predictLeaves[], const unsigned int bag[]) {
  // TODO:  Also catch mixed case in which no factors split, and avoid mixed case
  // in which no numericals split.
  if (nPredFac == 0)
    PredictAcrossNum(predictLeaves, bag);
  else if (nPredNum == 0) // Purely factor predictors.
    PredictAcrossFac(predictLeaves, bag);
  else  // Mixed numerical and factor
    PredictAcrossMixed(predictLeaves, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(int *leaves, const unsigned int bag[]) {
  double *transpose = new double[nPred * nRow];
  unsigned int row;

  // N.B.:  Parallelization by row assumes that nRow >> nTree.
  // TODO:  Consider blocking, to cut down on memory.  Mut. mut. for the
  // other two methods.
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowNum(row, &transpose[nPred * row], &leaves[nTree * row], bag);
  }
  }

  delete [] transpose;
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossFac(int *leaves, const unsigned int bag[]) {
  int *transpose = new int[nPred * nRow];
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowFac(row, &transpose[row * nPred], &leaves[row * nTree], bag);
  }
  }

  delete [] transpose;
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param prediction contains the mean score across trees.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(int *leaves, const unsigned int bag[]) {
  double *transposeN = new double[nPredNum * nRow];
  int *transposeI = new int[nPredFac * nRow];
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowMixed(row, &transposeN[row * nPredNum], &transposeI[row * nPredFac], &leaves[row * nTree], bag);
  }
  }

  delete [] transposeN;
  delete [] transposeI;
}


/**
   @brief Prediction for regression tree, with predictors of only numeric type.

   @param row is the row of data over which a prediction is made.

   @param rowT is a numeric data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */

void Forest::PredictRowNum(unsigned int row, double rowT[], int leaves[], const unsigned int bag[]) {
  for (int i = 0; i < nPred; i++)
    rowT[i] = Predictor::numBase[row + i* nRow];

  // TODO:  Use row at rank.
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }

    int tOrig = treeOrigin[tc];
    int *preds = pred + tOrig;
    double *splitVal = num + tOrig;
    int *bumps = bump + tOrig;

    int idx = 0;
    int inc = bumps[idx];
    while (inc != 0) {
      int pred = preds[idx];
      idx += (rowT[pred] <= splitVal[idx] ? inc : inc + 1);
      inc = bumps[idx];
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Prediction for regression tree, with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictRowFac(unsigned int row, int rowT[], int leaves[],  const unsigned int bag[]) {
  for (int i = 0; i < nPredFac; i++)
    rowT[i] = Predictor::facBase[row + i * nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int idx = 0;
    int tOrig = treeOrigin[tc];
    int *preds = pred + tOrig;
    double *splitVal = num + tOrig;
    int *bumps = bump + tOrig;
    int *fs = facSplit + facOff[tc];

    int inc = bumps[idx];
    while (inc != 0) {
      int facOff = int(splitVal[idx]);
      int pred = preds[idx];
      int facId = Predictor::FacIdx(pred);
      idx += (fs[facOff + rowT[facId]] ? inc : inc + 1);
      inc = bumps[idx];
    }
    leaves[tc] = idx;
    // TODO:  Instead of runtime check, can guarantee this by checking last level for non-negative
    // predictor fields.
  }
}


/**
   @brief Prediction for regression tree, with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictRowMixed(unsigned int row, double rowNT[], int rowFT[], int leaves[], const unsigned int bag[]) {
  for (int i = 0; i < nPredNum; i++)
    rowNT[i] = Predictor::numBase[row + i * nRow];
  for (int i = 0; i < nPredFac; i++)
    rowFT[i] = Predictor::facBase[row + i * nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int tOrig = treeOrigin[tc];
    int *preds = pred + tOrig;
    double *splitVal = num + tOrig;
    int *bumps = bump + tOrig;
    int *fs = facSplit + facOff[tc];

    int idx = 0;
    int inc = bumps[idx];
    while (inc != 0) {
      int pred = preds[idx];
      int facId = Predictor::FacIdx(pred);
      idx += (facId < 0 ? (rowNT[pred] <= splitVal[idx] ?  inc : inc + 1)  : (fs[int(splitVal[idx]) + rowFT[facId]] ? inc : inc + 1));
      inc = bumps[idx];
    }
    leaves[tc] = idx;
  }
  // TODO:  Instead of runtime check, can guarantee this by checking last level for non-negative
  // predictor fields.
}


