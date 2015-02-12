// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file dectree.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman

   These methods are mostly mechanical.  Several methods are tasked
   with populating or depopulating tree-related data structures.  The
   tree-walking methods are clones of one another, with slight variations
   based on response or predictor type.
 */


#include "predictor.h"
#include "dectree.h"
#include "response.h"
#include "pretree.h"
#include "quant.h"

#include <iostream>
using namespace std;

int *DecTree::treeOriginForest = 0; // Output to front-end.
int *DecTree::treeSizes = 0; // Internal use only.
int **DecTree::predTree = 0;
double **DecTree::splitTree = 0;
double **DecTree::scoreTree = 0;
int **DecTree::bumpTree = 0;

// Nonzero iff factor appear in decision tree.
//
int *DecTree::treeFacWidth = 0;
int **DecTree::treeFacSplits = 0;

int* DecTree::facSplitForest = 0; // Bits as integers:  alignment.
int *DecTree::facOffForest = 0;
int DecTree::nTree = -1;
int DecTree::nRow = -1;
int DecTree::nPred = -1;
int DecTree::nPredNum = -1;
int DecTree::nPredFac = -1;
double *DecTree::predInfo = 0;
int *DecTree::predForest = 0;
double *DecTree::scoreForest = 0;
double *DecTree::splitForest = 0;
int *DecTree::bumpForest = 0;
unsigned int *DecTree::inBag = 0;

int DecTree::forestSize = -1;

/**
   @brief Lights off the initializations for building decision trees.

   @param _nTree is the number of trees requested.

   @param _nRow is the number of samples in the response/observations.

   @param _nPred is the number of predictors.

   @param _nPredNum is the number of numeric predictors.

   @param _nPredFac is the number of factor-valued predictors.

   @param _doQuantiles inidicates whether quantile training is specified.

   @return void.

 */
void DecTree::FactoryTrain(int _nTree, int _nRow, int _nPred, int _nPredNum, int _nPredFac) {
  nTree = _nTree;
  nPred = _nPred;
  nRow = _nRow;
  nPredNum = _nPredNum;
  nPredFac = _nPredFac;
  forestSize = 0;
  treeOriginForest = new int[nTree];
  treeSizes = new int[nTree];
  predInfo = new double[nPred];
  predTree = new int*[nTree];
  splitTree = new double*[nTree];
  scoreTree = new double*[nTree];
  bumpTree = new int*[nTree];
  treeFacWidth = new int[nTree]; // Factor width counts of individual trees.
  treeFacSplits = new int* [nTree]; // Tree-based factor split values.
  for (int i = 0; i < nPred; i++)
    predInfo[i] = 0.0;
  for (int i = 0; i < nTree; i++)
    treeFacWidth[i] = 0;

  // Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  // may still prove too small for multi-gigarow sets.  Saving this state is
  // necessary, however, for per-row OOB prediction scheme employed for quantile
  // regression.
  //
  int inBagSize = ((nTree * nRow) + 31) >> 5;
  inBag = new unsigned int[inBagSize];
  for (int i = 0; i < inBagSize; i++)
    inBag[i] = 0;
}

/**
   @brief Loads trained forest from front end.

   @param _nTree is the number of trees in the forest.

   @param _forestSize is the length of the multi-vector holding all tree parameters.

   @param _preds[] are the predictors associated with tree nonterminals.

   @param _splits[] are the splitting values associated with nonterminals.

   @param _scores[] are the scores associated with terminals.

   @param _origins[] are the offsets into the multivector denoting each individual tree vector.

   @param _facOff[] are the offsets into the multi-bitvector denoting each tree's factor splitting values.

   @param _facSplits[] are the factor splitting values. 

   @return void.
*/

void DecTree::ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bump[], int _origins[], int _facOff[], int _facSplits[]) {
  nTree = _nTree;
  forestSize = _forestSize;
  predForest = _preds;
  splitForest = _splits;
  scoreForest = _scores;
  treeOriginForest = _origins;

  // Only used if categorical predictors present.
  //
  facOffForest = _facOff;
  facSplitForest = _facSplits;

  // Populates a packed table from two distinct vectors.
  bumpForest = new int[forestSize];
  for (int i = 0; i < forestSize; i++) {
    bumpForest[i] = _bump[i];
  }
}

/**
  @brief Resets addresses of vectors used during prediction.  Most are allocated
  by the front end so are not deallocated here.

  @return void
*/
void DecTree::DeFactoryPredict() {
  delete [] bumpForest; // Built on reload.
  bumpForest = 0;
  predForest = 0;
  splitForest = 0;
  scoreForest = 0;
  facSplitForest = 0;
  facOffForest = 0;
  forestSize = nTree = -1;

  Quant::DeFactoryPredict();

  Predictor::DeFactory();
}

/**
   @brief General deallocation after train/predict combination.

   @return void
 */
void DecTree::DeFactory() {
  delete [] treeSizes;
  delete [] treeOriginForest;
  delete [] predTree; // Contents deleted at consumption.
  delete [] splitTree; // "
  delete [] scoreTree; // "
  delete [] bumpTree; // "
  delete [] treeFacWidth;
  delete [] treeFacSplits; // Inidividual components deleted when tree written.
  delete [] inBag;
  delete [] predInfo;
  delete [] predForest;
  delete [] splitForest;
  delete [] scoreForest;
  delete [] bumpForest;
  delete [] facOffForest; // Always built, but may be all zeroes.
  if (facSplitForest != 0) // Not built if no splitting factors.
    delete [] facSplitForest;

  facOffForest = 0;
  facSplitForest = 0;

  treeSizes = 0;
  treeOriginForest = 0;
  predTree = 0;
  splitTree = 0;
  scoreTree = 0;
  bumpTree = 0;
  nTree = forestSize = -1;
  treeFacWidth = 0;
  treeFacSplits = 0;
  inBag = 0;

  bumpForest = 0;
  predForest = 0;
  splitForest = 0;
  scoreForest = predInfo =  0;
}

/**
   @brief Consumes remaining tree-based information into forest-wide data structures.

   @param cumFacWidth outputs the sum of all widths of factor bitvectors.

   @return Length of forest-wide vectors, plus output reference parameter.

 */
int DecTree::ConsumeTrees(int &cumFacWidth) {
  facOffForest = new int[nTree];

  cumFacWidth = 0;
  for (int tn = 0; tn < nTree; tn++) {
    facOffForest[tn] = cumFacWidth;
    cumFacWidth += treeFacWidth[tn];
  }

  if (cumFacWidth > 0) {
    facSplitForest = new int[cumFacWidth];

    int *facSplit = facSplitForest;
    for (int tn = 0; tn < nTree; tn++) {
      int fw = treeFacWidth[tn];
      if (fw > 0) {
	int *fs = treeFacSplits[tn];
	for (int i = 0; i < fw; i++) {
	  facSplit[i] = fs[i];
	}
	delete [] treeFacSplits[tn];
	treeFacSplits[tn] = 0;
	facSplit += fw;
      }
    }
  }

  predForest = new int[forestSize];
  splitForest = new double[forestSize];
  scoreForest = new double[forestSize];
  bumpForest = new int[forestSize];

  for (int i = 0; i < nTree; i++) {
    int start = treeOriginForest[i];
    for (int j = 0; j < treeSizes[i]; j++) {
      predForest[start + j] = predTree[i][j];
      splitForest[start + j] = splitTree[i][j];
      scoreForest[start + j] = scoreTree[i][j];
      bumpForest[start + j] = bumpTree[i][j];
    }
    delete [] predTree[i];
    delete [] splitTree[i];
    delete [] scoreTree[i];
    delete [] bumpTree[i];
  }
  Quant::ConsumeTrees(treeOriginForest, forestSize);

  return forestSize;
}

/**
  @brief Consumes pretree into per-tree data structures.

  @param _inBag enumerates the bagged rows for the current tree.

  @param bagCount is the number of bagged rows.

  @param treeSize is the number of nodes in this tree.

  @param treeNum is the zero-based tree number.

  @return void
*/
void DecTree::ConsumePretree(const bool _inBag[], int bagCount, int treeSize, int treeNum) {
  SetBagRow(_inBag, treeNum);
  treeSizes[treeNum] = treeSize;
  predTree[treeNum] = new int[treeSize];
  splitTree[treeNum] = new double[treeSize];
  bumpTree[treeNum] = new int[treeSize];
  scoreTree[treeNum] = new double[treeSize];

  // Employs data freed by pretree consumption, so must be called here.
  //
  Quant::TreeRanks(treeNum, treeSize, bagCount);

  // Consumes pretree nodes, ranks and split bits via separate calls.
  //
  PreTree::ConsumeNodes(leafPred, predTree[treeNum], splitTree[treeNum], bumpTree[treeNum], scoreTree[treeNum]);

  ConsumeSplitBits(treeNum);

  treeOriginForest[treeNum] = forestSize;
  forestSize += treeSize;
}

/**
 @brief Consumes splitting bitvector for the current pretree.

 @param treeNum is the index of the tree under constuction.

 @param facWidth is the count of splitting bits to be copied.

 @return void.
*/
void DecTree::ConsumeSplitBits(int treeNum) {
  int facWidth = PreTree::SplitFacWidth();
  treeFacWidth[treeNum] = facWidth;
  if (facWidth > 0) {
    treeFacSplits[treeNum] = new int[facWidth];
    PreTree::ConsumeSplitBits(treeFacSplits[treeNum]);
  }
  else
    treeFacSplits[treeNum] = 0;
}


/**
  @brief Sets bit for <row, tree> with tree as faster-moving index.

  @param sampledRow[]
  
  @return void.
*/
void DecTree::SetBagRow(const bool sampledRow[], int treeNum) {
  for (int row = 0; row < nRow; row++) {
    if (sampledRow[row]) {
      int idx = row * nTree + treeNum;
      int off = idx >> 5;
      int bit = idx & 31;
      unsigned int val = inBag[off];
      val |= (1 << bit);
      inBag[off] = val;
    }
  }
}

/**
   @brief Determines whether a given row index is in-bag in a given tree.

   @param treeNum is the index of a given tree.

   @param row is the row index to be tested.

   @return True iff the row is in-bag.
 */
bool DecTree::InBag(int treeNum, int row) {
  int idx = row * nTree + treeNum;
  int off = idx >> 5;
  int bit = idx & 31;
  unsigned int val = inBag[off];

  return (val & (1 << bit)) > 0;
}

void DecTree::WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBump, int *rOrigins, int *rFacOff, int * rFacSplits) {
  for (int tn = 0; tn < nTree; tn++) {
    int tOrig = treeOriginForest[tn];
    int facOrigin = facOffForest[tn];
    rOrigins[tn] = tOrig;
    rFacOff[tn] = facOrigin;
    WriteTree(tn, tOrig, facOrigin, rPreds + tOrig, rSplits + tOrig, rScores + tOrig, rBump + tOrig, rFacSplits + facOrigin);
  }
  DeFactory();
}


// Writes the tree-specific splitting information for export.
// Predictor indices are not written as 1-based indices.
//
void DecTree::WriteTree(int treeNum, int tOrig, int tFacOrig, int *outPreds, double* outSplitVals, double* outScores, int *outBump, int *outFacSplits) {
  int *preds = predForest + tOrig;
  double *splitVal = splitForest + tOrig;
  double *score = scoreForest + tOrig;
  int *bump = bumpForest + tOrig;

  for (int i = 0; i < treeSizes[treeNum]; i++) {
    outPreds[i] = preds[i]; // + 1;
    // Bumps the within-tree offset by cumulative tree offset so that tree is written
    // using absolute offsets within the global factor split vector.
    //
    // N.B.:  Both OOB and replay prediction use tree-relative offset numbers.
    //
    outSplitVals[i] = splitVal[i];
    outScores[i] = score[i];
    outBump[i] = bump[i];
  }

  // Even with factor predictors these could all be zero, as in the case of mixed predictor
  // types in which only the numerical predictors split.
  //
  int facWidth = treeFacWidth[treeNum];
  if (facWidth > 0) {
    int *facSplit = facSplitForest + tFacOrig;
    for (int i = 0; i < facWidth; i++)
      outFacSplits[i] = facSplit[i];
  }
}

/**
   @brief Scales the predictor Info values by the tree count.

   @param outPredInfo outputs the Info values.

   @return Formally void, with output parameter vector.
 */
void DecTree::ScaleInfo(double outPredInfo[]) {
  for (int i = 0; i < nPred; i++)
    outPredInfo[i] = predInfo[i] / nTree;
}

/**
   @brief Main driver for prediting categorical response.

   @param yCtg contains the training response, in the case of bagged prediction, otherwise the predicted response.

   @param ctgWidth is the cardinality of the response.

   @param confusion is an output confusion matrix.

   @param error is an output vector of classification errors.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictAcrossCtg(int yCtg[], int ctgWidth, int confusion[], double error[], bool useBag) {
  if (nPredFac == 0) {
    PredictAcrossNumCtg(yCtg, ctgWidth, confusion, useBag);
  }
  else if (nPredNum == 0) {
    PredictAcrossFacCtg(yCtg, ctgWidth, confusion, useBag);
  }
  else {
    PredictAcrossMixedCtg(yCtg, ctgWidth, confusion, useBag);
  }

  if (useBag) { // Otherwise, no test-vector against which to compare.
    // Fills in classification error vector.
    //
    for (int rsp = 0; rsp < ctgWidth; rsp++) {
      int numWrong = 0;
      for (int predicted = 0; predicted < ctgWidth; predicted++) {
	if (predicted != rsp) {// Wrong answers are off-diagonal.
	  numWrong += confusion[rsp + ctgWidth * predicted];
	}
      }
      error[rsp] = double(numWrong) / double(numWrong + confusion[rsp + ctgWidth * rsp]);
    }
  }
  else // Prediction only:  not training.
    DeFactoryPredict();
}

/**
   @brief Categorical prediction across rows with numerical predictor type.

   @param yCtg contains the training response, in the case of bagged prediction, otherwise the predicted response.

   @param ctgWidth is the cardinality of the response.

   @param confusion is an output confusion matrix.

   @param error is an output vector of classification errors.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictAcrossNumCtg(int yCtg[], int ctgWidth, int confusion[], bool useBag) {
  double *rowSlice = new double[nPred];
  int *rowPred = new int[ctgWidth];
  // TODO:  Parallelize.  There can be considerable false sharing if 'ctgWidth'
  // is small, so row-based approach may need to be reconsidered.
  // Mut. mut. for the other two methods.
  int row;
  for (row = 0; row < nRow; row++) {
    // double jitter = 1 + ResponseCtg::Jitter(row);
    PredictRowNumCtg(row, rowSlice, ctgWidth, rowPred, useBag);
    int argMax = -1;
    double popMax = 0.0;
    for (int col = 0; col < ctgWidth; col++) {
      int colPop = rowPred[col];// * jitter;
      if (colPop > popMax) {
	popMax = colPop;
	argMax = col;
      }
    }
    if (argMax >= 0) {
      if (useBag) {
	int rsp = yCtg[row];
	confusion[rsp + ctgWidth * argMax]++;
      }
      else
	yCtg[row] = argMax;
    }
  }

  delete [] rowSlice;
  delete [] rowPred;
}

/**
   @brief Categorical prediction across rows with factor predictor type.

   @param yCtg contains the training response, in the case of bagged prediction, otherwise the predicted response.

   @param ctgWidth is the cardinality of the response.

   @param confusion is an output confusion matrix.

   @param error is an output vector of classification errors.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictAcrossFacCtg(int yCtg[], int ctgWidth, int confusion[], bool useBag) {
  int *rowSlice = new int[nPred];
  int *rowPred = new int[ctgWidth];

  for (int row = 0; row < nRow; row++) {
    //double jitter = 1 + ResponseCtg::Jitter(row);
    PredictRowFacCtg(row, rowSlice, ctgWidth, rowPred, useBag);
    int argMax = -1;
    double popMax = 0.0;
    for (int col = 0; col < ctgWidth; col++) {
      double colPop = rowPred[col];// * jitter;
      if (colPop > popMax) {
	popMax = colPop;
	argMax = col;
      }
    }
    if (argMax >= 0) {
      if (useBag) {
	int rsp = yCtg[row];
	confusion[rsp + ctgWidth * argMax]++;
      }
      else
	yCtg[row] = argMax;
    }
  }
  delete [] rowSlice;
  delete [] rowPred;
}

/**
   @brief Categorical prediction across rows with mixed predictor types.

   @param yCtg contains the training response, in the case of bagged prediction, otherwise the predicted response.

   @param ctgWidth is the cardinality of the response.

   @param confusion is an output confusion matrix.

   @param error is an output vector of classification errors.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictAcrossMixedCtg(int yCtg[], int ctgWidth, int confusion[], bool useBag) {
  int *rowPred = new int[ctgWidth];
  double *rowSliceN = new double[nPredNum];
  int *rowSliceI = new int[nPredFac];

  for (int row = 0; row < nRow; row++) {
    //double jitter = 1 + ResponseCtg::Jitter(row);
    PredictRowMixedCtg(row, rowSliceN, rowSliceI, ctgWidth, rowPred, useBag);
    int argMax = -1;
    double popMax = 0.0;
    for (int col = 0; col < ctgWidth; col++) {
      double colPop = rowPred[col];// * jitter;
      if (colPop > popMax) {
	popMax = colPop;
	argMax = col;
      }
    }
    if (argMax >= 0) {
      if (useBag) {
	int rsp = yCtg[row];
	confusion[rsp + ctgWidth *argMax]++;
      }
      else
	yCtg[row] = argMax;
    }
  }
  delete [] rowSliceN;
  delete [] rowSliceI;
  delete [] rowPred;
}

/**
   @brief Main driver for prediting regression response.

   @param outVec contains the predictions.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictAcrossReg(double outVec[], bool useBag) {
  double *prediction;

  if (useBag)
    prediction = new double[nRow];
  else
    prediction = outVec;
  int *predictLeaves = new int[nTree * nRow];

  // TODO:  Also catch mixed case in which no factors split, and avoid mixed case
  // in which no numericals split.
  if (nPredFac == 0)
    PredictAcrossNumReg(prediction, predictLeaves, useBag);
  else if (nPredNum == 0) // Purely factor predictors.
    PredictAcrossFacReg(prediction, predictLeaves, useBag);
  else  // Mixed numerical and factor
    PredictAcrossMixedReg(prediction, predictLeaves, useBag);

  Quant::PredictRows(treeOriginForest, predictLeaves);
  delete [] predictLeaves;

  if (useBag) {
    double SSE = 0.0;
    for (int row = 0; row < nRow; row++) {
      SSE += (prediction[row] - Response::response->y[row]) * (prediction[row] - Response::response->y[row]);
    }
    // TODO:  repair assumption that every row is sampled:
    // Assumes nonzero nRow:
    outVec[0] = SSE / nRow;

    delete [] prediction;
  }
  else
    DeFactoryPredict();
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param prediction contains the mean score across trees.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */

void DecTree::PredictAcrossNumReg(double prediction[], int *predictLeaves, bool useBag) {
  double *transpose = new double[nPred * nRow];
  int row;

  // N.B.:  Parallelization by row assumes that nRow >> nTree.
  // TODO:  Consider blocking, to cut down on memory.  Mut. mut. for the
  // other two methods.
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    int *leaves = predictLeaves + row * nTree;
    double *rowSlice = transpose + row * nPred;
    PredictRowNumReg(row, rowSlice, leaves, useBag);
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leaves[tc];
      if (leafIdx >= 0) {
	treesSeen++;
	score +=  *(scoreForest + treeOriginForest[tc] + leafIdx);
      }
    }
    prediction[row] = score / treesSeen; // Assumes >= 1 tree seen.
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
void DecTree::PredictAcrossFacReg(double prediction[], int *predictLeaves, bool useBag) {
  int *transpose = new int[nPred * nRow];
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    int *leaves = predictLeaves + row * nTree;
    int *rowSlice = transpose + row * nPred;
    PredictRowFacReg(row, rowSlice, leaves, useBag);
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leaves[tc];
      if (leafIdx >= 0) {
	treesSeen++;
	score +=  *(scoreForest + treeOriginForest[tc] + leafIdx);
      }
    }
    prediction[row] = score / treesSeen; // Assumes >= 1 tree seen.
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
void DecTree::PredictAcrossMixedReg(double prediction[], int *predictLeaves, bool useBag) {
  double *transposeN = new double[nPredNum * nRow];
  int *transposeI = new int[nPredFac * nRow];
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    double *rowSliceN = transposeN + row * nPredNum;
    int *rowSliceI = transposeI + row * nPredFac;
    int *leaves = predictLeaves + row * nTree;
    PredictRowMixedReg(row, rowSliceN, rowSliceI, leaves, useBag);
    for (int tc = 0; tc < nTree; tc++) {
      int leafIdx = leaves[tc];
      if (leafIdx >= 0) {
        treesSeen++;
        score +=  *(scoreForest + treeOriginForest[tc] + leafIdx);
      }
    }
    prediction[row] = score / treesSeen; // Assumes >= 1 tree seen.
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

void DecTree::PredictRowNumReg(int row, double rowT[], int leaves[], bool useBag) {
  for (int i = 0; i < nPred; i++)
    rowT[i] = Predictor::numBase[row + i* nRow];

  // TODO:  Use row at rank.
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int idx = 0;
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    int *bumps = bumpForest + tOrig;

    int pred = preds[idx];
    while (pred != leafPred) {
      int bump = bumps[idx];
      idx += (rowT[pred] <= splitVal[idx] ? bump : bump + 1);
      pred = preds[idx];
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Prediction for classification tree, with predictors of only numeric type.

   @param row is the row of data over which a prediction is made.

   @param rowT is a numeric data array section corresponding to the row.

   @param prd[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */

void DecTree::PredictRowNumCtg(int row, double rowT[], int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < nPred; i++)
    rowT[i] = Predictor::numBase[row + i* nRow];
  for (int i = 0; i < ctgWidth; i++)
    prd[i] = 0;

  // TODO:  Use row at rank.
  //int ct = 0;
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row))
      continue;
    int idx = 0;
    int tOrig = treeOriginForest[tc];
    //ct++;
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    int *bumps = bumpForest + tOrig;

    int pred = preds[idx];
    while (pred != leafPred) {
      int bump = bumps[idx];
      idx += (rowT[pred] <= splitVal[idx] ? bump : bump + 1);
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
  }
}

// Temporary clone of regression tree version.
//
void DecTree::PredictRowFacCtg(int row, int rowT[], int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < nPred; i++)
    rowT[i] = Predictor::facBase[row + i* nRow];

  for (int i = 0; i < ctgWidth; i++)
    prd[i] = 0;

  // TODO:  Use row at rank.

  for (int tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row))
      continue;
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    int *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      int bump = bumps[idx];
      int facOff = int(splitVal[idx]);
      int facId = Predictor::FacIdx(pred);
      idx += (fs[facOff + rowT[facId]] ? bump : bump + 1);
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
  }
}

/**
   @brief Prediction for classification tree, with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param ctgWidth is the cardinality of the response.

   @param prd[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void DecTree::PredictRowMixedCtg(int row, double rowNT[], int rowFT[], int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < nPredNum; i++)
    rowNT[i] = Predictor::numBase[row + i * nRow];
  for (int i = 0; i < nPredFac; i++)
    rowFT[i] = Predictor::facBase[row + i * nRow];

  for (int i = 0; i < ctgWidth; i++)
    prd[i] = 0;

  for (int tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row))
      continue;
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    int *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      int bump = bumps[idx];
      int facId = Predictor::FacIdx(pred);
      idx += (facId < 0 ? (rowNT[pred] <= splitVal[idx] ?  bump : bump + 1) : (fs[int(splitVal[idx]) + rowFT[facId]] ? bump : bump + 1));
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
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

void DecTree::PredictRowFacReg(int row, int rowT[], int leaves[],  bool useBag) {
  for (int i = 0; i < nPredFac; i++)
    rowT[i] = Predictor::facBase[row + i * nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int idx = 0;
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    int *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int pred = preds[idx];
    while (pred != leafPred) {
      int facOff = int(splitVal[idx]);
      int bump = bumps[idx];
      int facId = Predictor::FacIdx(pred);
      idx += (fs[facOff + rowT[facId]] ? bump : bump + 1);
      pred = preds[idx];
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
void DecTree::PredictRowMixedReg(int row, double rowNT[], int rowFT[], int leaves[], bool useBag) {
  for (int i = 0; i < nPredNum; i++)
    rowNT[i] = Predictor::numBase[row + i * nRow];
  for (int i = 0; i < nPredFac; i++)
    rowFT[i] = Predictor::facBase[row + i * nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    int *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      int bump = bumps[idx];
      int facId = Predictor::FacIdx(pred);
      idx += (facId < 0 ? (rowNT[pred] <= splitVal[idx] ?  bump : bump + 1)  : (fs[int(splitVal[idx]) + rowFT[facId]] ? bump : bump + 1));
      pred = preds[idx];
    }
    leaves[tc] = idx;
  }
  // TODO:  Instead of runtime check, can guarantee this by checking last level for non-negative
  // predictor fields.
}
