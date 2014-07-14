/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "predictor.h"
#include "train.h"
#include "level.h"
#include "splitsig.h"
#include "dectree.h"
#include "response.h"
#include "predict.h"
#include "pretree.h"

#include <iostream>
using namespace std;

int *DecTree::treeOriginForest = 0; // Output to front-end.
int *DecTree::treeSizes = 0; // Internal use only.
int **DecTree::treePreds = 0;
double **DecTree::treeSplits = 0;
double **DecTree::treeScores = 0;
Bump **DecTree::treeBumps = 0;

// Nonzero iff quantiles stipulated for training.
int *DecTree::treeQRankWidth = 0;
int *DecTree::treeQLeafWidth = 0;
int **DecTree::treeQLeafPos = 0;
int **DecTree::treeQLeafExtent = 0;
int **DecTree::treeQRank = 0;
int **DecTree::treeQRankCount = 0;

// Nonzero iff quantiles tabulated.
double *DecTree::qYRankedForest = 0;
int DecTree::qYLenForest = -1;
int *DecTree::qRankOriginForest = 0;
int *DecTree::qRankForest = 0;
int *DecTree::qRankCountForest = 0;
int *DecTree::qLeafPosForest = 0;
int *DecTree::qLeafExtentForest = 0;

// Nonzero iff factor appear in decision tree.
//
int *DecTree::treeFacWidth = 0;
int **DecTree::treeFacSplits = 0;

int* DecTree::facSplitForest = 0; // Bits as integers:  alignment.
int *DecTree::facOffForest = 0;
double DecTree::recipNumTrees = 0.0;
int DecTree::nTree = -1;
double *DecTree::predGini = 0;
int *DecTree::predForest = 0;
double *DecTree::scoreForest = 0;
double *DecTree::splitForest = 0;
Bump *DecTree::bumpForest = 0;
unsigned int *DecTree::inBag = 0;

int DecTree::forestSize = -1;
int DecTree::totBagCount = -1;
int DecTree::totQLeafWidth = -1;  // Size of compressed quantile leaf vector.

void DecTree::ForestTrain(const int _nTree) {
  nTree = _nTree;
  forestSize = 0;
  totBagCount = 0;
  treeOriginForest = new int[nTree];
  treeSizes = new int[nTree];
  nTree = _nTree;
  recipNumTrees = 1.0 / nTree;
  predGini = new double[Predictor::nPred];
  treePreds = new int*[nTree];
  treeSplits = new double*[nTree];
  treeScores = new double*[nTree];
  treeBumps = new Bump*[nTree];
  treeFacWidth = new int[nTree]; // Factor width counts of individual trees.
  treeFacSplits = new int* [nTree]; // Tree-based factor split values.
  for (int i = 0; i < Predictor::nPred; i++)
    predGini[i] = 0.0;
  for (int i = 0; i < nTree; i++)
    treeFacWidth[i] = 0;

  // Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  // may still prove too small for multi-gigarow sets.  Saving this state is
  // necessary, however, for per-row OOB prediction scheme employed for quantile
  // regression.
  //
  int inBagSize = ((nTree * Predictor::nRow) + 31) >> 5;
  inBag = new unsigned int[inBagSize];
  for (int i = 0; i < inBagSize; i++)
    inBag[i] = 0;

  if (Train::doQuantiles) { // Tree-based quantile data structures.
    treeQRankWidth = new int[nTree];
    treeQLeafWidth = new int[nTree];
    treeQLeafPos = new int*[nTree];
    treeQLeafExtent = new int*[nTree];
    treeQRank = new int*[nTree];
    treeQRankCount = new int*[nTree];

    for (int i = 0; i < nTree; i++) { // Actually not necessary, as all trees have >= 1 leaf.
      treeQRankWidth[i] = 0;
      treeQLeafWidth[i] = 0;
      treeQLeafPos[i] = 0;
      treeQLeafExtent[i] = 0;
      treeQRank[i] = 0;
    }
  }
}

// Reloads cached tree data from front end.
//
void DecTree::ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bumpL[], int _bumpR[], int _origins[], int _facOff[], int _facSplits[]) {
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
  bumpForest = new Bump[forestSize];
  for (int i = 0; i < forestSize; i++) {
    bumpForest[i].left = _bumpL[i];
    bumpForest[i].right = _bumpR[i];
  }
}

// Reloads cached quantile data from front end.
void DecTree::ForestReloadQuant(double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]) {

  qYRankedForest = qYRanked;
  qYLenForest = qYLen;
  qRankOriginForest = qRankOrigin;
  qRankForest = qRank;
  qRankCountForest = qRankCount;
  qLeafPosForest = qLeafPos;
  qLeafExtentForest= qLeafExtent;
}

void DecTree::WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]) {
  for (int i = 0; i < Predictor::nRow; i++)
    rQYRanked[i] = qYRankedForest[i];

  for (int tn = 0; tn < nTree; tn++)
    rQRankOrigin[tn] = qRankOriginForest[tn];

  for (int i = 0; i < totBagCount; i++) {
    rQRank[i] = qRankForest[i];
    rQRankCount[i] = qRankCountForest[i];
  }
  for (int i = 0; i < totQLeafWidth; i++) {
    rQLeafPos[i] = qLeafPosForest[i];
    rQLeafExtent[i] = qLeafExtentForest[i];
  }
}

// Most of the vectors and matrices referenced here are passed in from R and should
// not be deleted.
//
void DecTree::DeForestPredict() {
  delete [] bumpForest; // Built on reload.
  bumpForest = 0;
  predForest = 0;
  splitForest = 0;
  scoreForest = 0;
  facSplitForest = 0;
  facOffForest = 0;
  qYRankedForest = 0;
  qRankOriginForest = qRankForest = qLeafPosForest = qLeafExtentForest = 0;
  qYLenForest = forestSize = totBagCount = nTree = -1;
}

void DecTree::DeForest() {
  delete [] treeSizes;
  delete [] treeOriginForest;
  delete [] treePreds; // Contents deleted at consumption.
  delete [] treeSplits; // "
  delete [] treeScores; // "
  delete [] treeBumps; // "
  delete [] treeFacWidth;
  delete [] treeFacSplits; // Inidividual components deleted when tree written.
  delete [] inBag;
  delete [] predGini;
  delete [] predForest;
  delete [] splitForest;
  delete [] scoreForest;
  delete [] bumpForest;
  delete [] facOffForest; // Always built, but may be all zeroes.
  if (facSplitForest != 0) // Not built if no splitting factors.
    delete [] facSplitForest;

  if (Train::doQuantiles) {
    delete [] qYRankedForest;
    delete [] qRankOriginForest;
    delete [] qRankForest;
    delete [] qRankCountForest;
    delete [] qLeafPosForest;
    delete [] qLeafExtentForest;
  }
  facOffForest = 0;
  facSplitForest = 0;

  qYRankedForest = 0;
  qRankOriginForest = 0;
  qRankForest = 0;
  qRankCountForest = 0;
  qLeafPosForest = 0;
  qLeafExtentForest = 0;

  treeSizes = 0;
  treeOriginForest = 0;
  treePreds = 0;
  treeSplits = 0;
  treeScores = 0;
  treeBumps = 0;
  recipNumTrees = 0.0;
  nTree = forestSize = qYLenForest = totBagCount = -1;
  treeFacWidth = 0;
  treeFacSplits = 0;
  inBag = 0;

  // Tree-based quantile data.
  treeQRankWidth = 0;
  treeQLeafWidth = 0;
  treeQLeafPos = 0;
  treeQLeafExtent = 0;
  treeQRank = 0;
  treeQRankCount = 0;;

  bumpForest = 0;
  predForest = 0;
  splitForest = 0;
  scoreForest = predGini =  0;
  qRankOriginForest = qLeafPosForest = qRankForest = 0;
}


void DecTree::ConsumePretree(const bool _inBag[], const int levels, const int treeNum) {
  SetBagRow(_inBag, treeNum);
  int treeSize = PreTree::TreeOffsets(levels);
  // TODO:  Consider uninitialized slots.
  treeSizes[treeNum] = treeSize;
  treePreds[treeNum] = new int[treeSize];
  treeSplits[treeNum] = new double[treeSize];
  treeScores[treeNum] = new double[treeSize];
  treeBumps[treeNum] = new Bump[treeSize];
  //cout << "Tree " << treeNum << ": " << treeSize << endl;

  if (Train::doQuantiles)
    QuantileRanks(treeNum, treeSize, PreTree::bagCount);

  PreTree::ConsumeLeaves(treeScores[treeNum], treePreds[treeNum]);
  ConsumeSplits(treeNum, treeSplits[treeNum], treePreds[treeNum], treeBumps[treeNum]);

  totBagCount += PreTree::bagCount;
  treeOriginForest[treeNum] = forestSize;
  forestSize += treeSize;
}

int DecTree::AllTrees(int *cumFacWidth, int *cumBagWidth, int *cumQLeafWidth) {
  facOffForest = new int[nTree];

  // Returns sum of factor widths so that R caller can allocate vector holding all
  // splitting values.
  //
  int totFacWidth = 0;
  for (int tn = 0; tn < nTree; tn++) {
    facOffForest[tn] = totFacWidth;
    totFacWidth += treeFacWidth[tn];
  }
  *cumFacWidth = totFacWidth;

  if (totFacWidth > 0) {
    facSplitForest = new int[totFacWidth];

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

  if (Train::doQuantiles) {
    int totQRankWidth = 0;
    totQLeafWidth = 0;

    qYRankedForest = new double[Predictor::nRow];
    qYLenForest = Predictor::nRow;
    Response::response->GetYRanked(qYRankedForest);

    qRankOriginForest = new int[nTree];
    for (int tn = 0; tn < nTree; tn++) {
      qRankOriginForest[tn] = totQRankWidth;
      totQRankWidth += treeQRankWidth[tn]; // bagCount
      totQLeafWidth += treeQLeafWidth[tn];
    }

    qRankForest = new int[totQRankWidth];
    qRankCountForest = new int[totQRankWidth];
    qLeafPosForest = new int[totQLeafWidth];
    qLeafExtentForest = new int[totQLeafWidth];
    for (int tn = 0; tn < nTree; tn++) {
      int *qRank = qRankForest + qRankOriginForest[tn];
      int *qRankCount = qRankCountForest + qRankOriginForest[tn];
      for (int i = 0; i < treeQRankWidth[tn]; i++) {
	qRank[i] = treeQRank[tn][i];
	qRankCount[i] = treeQRankCount[tn][i];
      }
      delete [] treeQRank[tn];
      delete [] treeQRankCount[tn];

      int *qLeafPos = qLeafPosForest + treeOriginForest[tn];
      int *qLeafExtent = qLeafExtentForest + treeOriginForest[tn];
      for (int i = 0; i < treeQLeafWidth[tn]; i++) {
	qLeafPos[i] = treeQLeafPos[tn][i];
	qLeafExtent[i] = treeQLeafExtent[tn][i];
      }
      delete [] treeQLeafPos[tn];
      delete [] treeQLeafExtent[tn];
    }
    delete [] treeQRankWidth;
    delete [] treeQLeafWidth;
    delete [] treeQRank;
    delete [] treeQRankCount;
    delete [] treeQLeafPos;
    delete [] treeQLeafExtent;
    *cumQLeafWidth = totQLeafWidth;
  }
  else
    *cumQLeafWidth = 0;
  
  predForest = new int[forestSize];
  splitForest = new double[forestSize];
  scoreForest = new double[forestSize];
  bumpForest = new Bump[forestSize];
  //  for (int i = 0; i < forestSize; i++) // TODO:  Necessary?
	//predForest[i] = leafPred;

  // Consumes individual trees into forest-wide vectors.
  //
  for (int i = 0; i < nTree; i++) {
    int start = treeOriginForest[i];
    int treeSize = treeSizes[i];
    for (int j = 0; j < treeSize; j++) {
      int pred = treePreds[i][j];
      predForest[start + j] = pred;
      double split = treeSplits[i][j];
      splitForest[start + j] = split;
      scoreForest[start + j] = treeScores[i][j];
      bumpForest[start + j] = treeBumps[i][j];
    }
    delete [] treePreds[i];
    delete [] treeSplits[i];
    delete [] treeScores[i];
    delete [] treeBumps[i];
  }

  *cumBagWidth = totBagCount;
  return forestSize;
}


// Fills in the splitting information columns for the next tree. 'outGini' not used for training, and
// is written for diagnostic purposes.
//
void DecTree::ConsumeSplits(int treeNum, double splitVec[], int predVec[], Bump bumpVec[]) { //, double *outGini) {
  PreTree::ConsumeSplits(splitVec, predVec, bumpVec);
  if (Predictor::nPredFac > 0) {
    // Collects factor splitting from the various levels into single vector for the
    // entire tree.
    //
    int fw = SplitSigFac::SplitFacWidth();
    treeFacWidth[treeNum] = fw;
    if (fw > 0) {
      treeFacSplits[treeNum] = new int[fw];
      SplitSigFac::ConsumeTreeSplitBits(treeFacSplits[treeNum]);
    }
    else
      treeFacSplits[treeNum] = 0;
  }
}
    //    cout << "Tree bits " << treeNum << endl;
    //for (int i = 0; i < fw; i++)
    //cout << treeFacSplits[treeNum][i];


// Transfers quantile data structures from pretree to decision tree.
// TODO:  Exit treeQLeafWidth, as size is known.
void DecTree::QuantileRanks(const int tn, const int treeSize, const int bagCount) {
  treeQRankWidth[tn] = bagCount;
  treeQLeafWidth[tn] = treeSize;

  int *qLeafPos = new int[treeSize];
  treeQLeafPos[tn] = qLeafPos;
  int *qLeafExtent = new int[treeSize];
  treeQLeafExtent[tn] = qLeafExtent;

  int *qRank  = new int[bagCount];
  treeQRank[tn] = qRank;
  int *qRankCount = new int[bagCount];
  treeQRankCount[tn] = qRankCount;

  PreTree::DispatchQuantiles(treeSize, qLeafPos, qLeafExtent, qRank, qRankCount);
}

// Sets bit for <row, tree> with tree as faster-moving index.
//
void DecTree::SetBagRow(const bool sampledRow[], const int treeNum) {
  for (int row = 0; row < Predictor::nRow; row++) {
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

bool DecTree::InBag(int treeNum, int row) {
  int idx = row * nTree + treeNum;
  int off = idx >> 5;
  int bit = idx & 31;
  unsigned int val = inBag[off];

  return (val & (1 << bit)) > 0;
}

void DecTree::WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBumpL, int *rBumpR, int *rOrigins, int *rFacOff, int * rFacSplits) {
  for (int tn = 0; tn < nTree; tn++) {
    int tOrig = treeOriginForest[tn];
    int facOrigin = facOffForest[tn];
    rOrigins[tn] = tOrig;
    rFacOff[tn] = facOrigin;
    WriteTree(tn, tOrig, facOrigin, rPreds + tOrig, rSplits + tOrig, rScores + tOrig, rBumpL + tOrig, rBumpR + tOrig, rFacSplits + facOrigin);
  }
  DeForest();
}


// Writes the tree-specific splitting information for export.
// Predictor indices are not written as 1-based indices.
//
void DecTree::WriteTree(const int treeNum, const int tOrig, const int tFacOrig, int *outPreds, double* outSplitVals, double* outScores, int *outBumpL, int *outBumpR, int *outFacSplits) {
  int *preds = predForest + tOrig;
  double *splitVal = splitForest + tOrig;
  double *score = scoreForest + tOrig;
  Bump *bump = bumpForest + tOrig;

  for (int i = 0; i < treeSizes[treeNum]; i++) {
    outPreds[i] = preds[i]; // + 1;
    // Bumps the within-tree offset by cumulative tree offset so that tree is written
    // using absolute offsets within the global factor split vector.
    //
    // N.B.:  Both OOB and replay prediction use tree-relative offset numbers.
    //
    outSplitVals[i] = splitVal[i];
    outScores[i] = score[i];
    outBumpL[i] = bump[i].left;
    outBumpR[i] = bump[i].right;
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

void DecTree::ScaleGini(double *outPredGini) {
  for (int i = 0; i < Predictor::nPred; i++)
    outPredGini[i] = predGini[i] * recipNumTrees;
}

// Returns confusion matrix for OOB categorical predictions.
//
void DecTree::PredictAcrossCtg(int yCtg[], const int ctgWidth, int confusion[], double error[], bool useBag) {
  int numWrong = 0;
  int numTried = 0;
  int *rowPred = new int[ctgWidth];
  // Purely numerical predictors.
  if (Predictor::nPredFac == 0) {
    double *rowSlice = new double[Predictor::nPred];
    for (int row = 0; row < Predictor::nRow; row++) {
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
	// Deep prediction.
	//	confusion[off] += colPop;
	//numWrong += rsp == col ? 0 : colPop;
	//numTried += colPop;
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
  }
  else if (Predictor::nPredNum == 0) {
    int *rowSlice = new int[Predictor::nPred];
    for (int row = 0; row < Predictor::nRow; row++) {
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
	// Deep prediction.
	//	confusion[off] += colPop;
	//numWrong += rsp == col ? 0 : colPop;
	//numTried += colPop;
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
  }
  else {
    double *rowSliceN = new double[Predictor::nPredNum];
    int *rowSliceI = new int[Predictor::nPredFac];
    for (int row = 0; row < Predictor::nRow; row++) {
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
	// Deep prediction.
	//	confusion[off] += colPop;
	//numWrong += rsp == col ? 0 : colPop;
	//numTried += colPop;
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
  delete [] rowPred;
}

// Parametrize by 'useBag'.  If true, 'outVec' receives OOB error, otherwise the prediction.
//
void DecTree::PredictAcrossReg(double outVec[], bool useBag) {
  double *prediction;
  if (useBag)
    prediction = new double[Predictor::nRow];
  else
    prediction = outVec;

  // TODO:  Also catch mixed case in which no factors split, and avoid mixed case
  // in which no numericals split.
  if (Predictor::nPredFac == 0)
    PredictAcrossNumReg(prediction, useBag);
  else if (Predictor::nPredNum == 0) // Purely factor predictors.
    PredictAcrossFacReg(prediction, useBag);
  else  // Mixed numerical and factor
    PredictAcrossMixedReg(prediction, useBag);

  if (useBag) {
    double SSE = 0.0;
    for (int row = 0; row < Predictor::nRow; row++) {
      SSE += (prediction[row] - Response::response->y[row]) * (prediction[row] - Response::response->y[row]);
    }
    // TODO:  repair assumption that every row is sampled:
    // Assumes nonzero nRow:
    outVec[0] = SSE / Predictor::nRow;

    delete [] prediction;
  }
}

void DecTree::PredictAcrossNumReg(double prediction[], bool useBag) {
  double *transpose = new double[Predictor::nPred * Predictor::nRow];
  int *predictLeaves = new int[nTree * Predictor::nRow];
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < Predictor::nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    int *leaves = predictLeaves + row * nTree;
    double *rowSlice = transpose + row * Predictor::nPred;
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
  if (QuantSig::qCells > 0) {
    for (row = 0; row < Predictor::nRow; row++) {
      double *qRow = QuantSig::qPred + row * QuantSig::qCells;
      int *leaves = predictLeaves + row * nTree;
      QuantileLeaves(qRow, QuantSig::qCells, leaves);
    }
  }

  delete [] transpose;
  delete [] predictLeaves;
}

void DecTree::PredictAcrossFacReg(double prediction[], bool useBag) {
  int *transpose = new int[Predictor::nPred * Predictor::nRow];
  int *predictLeaves = new int[nTree * Predictor::nRow];
  int row;

  /*{
    for (int i = 0; i < nTree; i++) {
      cout << "Tree bits " << i << endl;
      for (int j = 0; j < treeFacWidth[i]; j++)
	cout << treeFacSplits[i][j];
      cout << endl;
      }*/

  for (row = 0; row < Predictor::nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    int *leaves = predictLeaves + row * nTree;
    int *rowSlice = transpose + row * Predictor::nPred;
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
  if (QuantSig::qCells > 0) {
    for (row = 0; row < Predictor::nRow; row++) {
      double *qRow = QuantSig::qPred + row * QuantSig::qCells;
      int *leaves = predictLeaves + row * nTree;
      QuantileLeaves(qRow, QuantSig::qCells, leaves);
    }
  }

  delete [] transpose;
  delete [] predictLeaves;
}

void DecTree::PredictAcrossMixedReg(double prediction[], bool useBag) {
  double *transposeN = new double[Predictor::nPredNum * Predictor::nRow];
  int *transposeI = new int[Predictor::nPredFac * Predictor::nRow];
  int *predictLeaves = new int[nTree * Predictor::nRow];
  int row;

  for (row = 0; row < Predictor::nRow; row++) {
    double score = 0.0;
    int treesSeen = 0;
    double *rowSliceN = transposeN + row * Predictor::nPredNum;
    int *rowSliceI = transposeI + row * Predictor::nPredFac;
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
  
  if (QuantSig::qCells > 0) {
    for (row = 0; row < Predictor::nRow; row++) {
      double *qRow = QuantSig::qPred + row * QuantSig::qCells;
      int *leaves = predictLeaves + row * nTree;
      QuantileLeaves(qRow, QuantSig::qCells, leaves);
    }
  }

  delete [] transposeN;
  delete [] transposeI;
  delete [] predictLeaves;
}

// Data matrix 'x'.  For passed 'row' selects predictor at lowest splitting level.
// Returns mean prediction across all trees in which 'row' appears out-of-bag.
//
void DecTree::PredictRowNumReg(const int row, double rowT[], int leaves[], bool useBag) {
  for (int i = 0; i < Predictor::nPred; i++)
    rowT[i] = Predictor::numBase[row + i* Predictor::nRow];

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
    Bump *bumps = bumpForest + tOrig;

    int pred = preds[idx];
    while (pred != leafPred) {
      Bump bump = bumps[idx];
      idx += (rowT[pred] <= splitVal[idx] ? bump.left : bump.right);
      pred = preds[idx];
    }
    leaves[tc] = idx;
  }
}


// Temporary clone of regression tree version.
//
void DecTree::PredictRowNumCtg(const int row, double rowT[], const int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < Predictor::nPred; i++)
    rowT[i] = Predictor::numBase[row + i* Predictor::nRow];
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
    Bump *bumps = bumpForest + tOrig;

    int pred = preds[idx];
    while (pred != leafPred) {
      Bump bump = bumps[idx];
      idx += (rowT[pred] <= splitVal[idx] ? bump.left : bump.right);
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
  }
}

// Temporary clone of regression tree version.
//
void DecTree::PredictRowFacCtg(const int row, int rowT[], const int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < Predictor::nPred; i++)
    rowT[i] = Predictor::facBase[row + i* Predictor::nRow];

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
    Bump *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      pred = -(pred+1);
      Bump bump = bumps[idx];
      int facOff = int(splitVal[idx]);
      //      cout << idx << ": " << pred << ", " << facOff << " " << facOffForest[tc] << endl;
      idx += (fs[facOff + rowT[pred]] ? bump.left : bump.right);
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
  }
}

// Data matrix 'x'.  For passed 'row' selects predictor at lowest splitting level.
// Returns mean prediction across all trees in which 'row' appears out-of-bag.
//
void DecTree::PredictRowMixedCtg(const int row, double rowNT[], int rowFT[], const int ctgWidth, int prd[], bool useBag) {
  for (int i = 0; i < Predictor::nPredNum; i++)
    rowNT[i] = Predictor::numBase[row + i * Predictor::nRow];
  for (int i = 0; i < Predictor::nPredFac; i++)
    rowFT[i] = Predictor::facBase[row + i * Predictor::nRow];

  for (int i = 0; i < ctgWidth; i++)
    prd[i] = 0;

  for (int tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row))
      continue;
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    Bump *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      //      cout << idx << endl;
      //if (pred >= 0)
      //cout << "\tN:  " << pred << ", " << rowNT[pred] << ", " << splitVal[idx] << endl;
      //else
      //cout << "\tF:  " << -(pred+1) << ", " << facOff << ", " << rowFT[-(pred+1)] << ", " << fs[facOff+rowFT[-(pred+1)]] <<endl;
      Bump bump = bumps[idx];
      idx += (pred >= 0 ? (rowNT[pred] <= splitVal[idx] ?  bump.left : bump.right) : (fs[int(splitVal[idx]) + rowFT[-(pred+1)]] ? bump.left : bump.right));
      pred = preds[idx];
    }
    int ctgPredict = scores[idx];
    prd[ctgPredict]++;
  }
}

// Data matrix 'x'.  For passed 'row' selects predictor at lowest splitting level.
// Returns mean prediction across all trees in which 'row' appears out-of-bag.
//
void DecTree::PredictRowFacReg(const int row, int rowT[], int leaves[],  bool useBag) {
  for (int i = 0; i < Predictor::nPredFac; i++)
    rowT[i] = Predictor::facBase[row + i * Predictor::nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int idx = 0;
    int tOrig = treeOriginForest[tc];
    // TODO:  Pure factor trees can be made to use positive values, on writing.
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    Bump *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int pred = preds[idx];
    while (pred != leafPred) {
      pred = -(pred + 1);
      int facOff = int(splitVal[idx]);
      Bump bump = bumps[idx];
      idx += (fs[facOff + rowT[pred]] ? bump.left : bump.right);
      pred = preds[idx];
    }
    leaves[tc] = idx;
    // TODO:  Instead of runtime check, can guarantee this by checking last level for non-negative
    // predictor fields.
  }
}

// Data matrix 'x'.  For passed 'row' selects predictor at lowest splitting level.
// Returns mean prediction across all trees in which 'row' appears out-of-bag.
//
void DecTree::PredictRowMixedReg(const int row, double rowNT[], int rowFT[], int leaves[], bool useBag) {
  for (int i = 0; i < Predictor::nPredNum; i++)
    rowNT[i] = Predictor::numBase[row + i * Predictor::nRow];
  for (int i = 0; i < Predictor::nPredFac; i++)
    rowFT[i] = Predictor::facBase[row + i * Predictor::nRow];

  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (useBag && InBag(tc, row)) {
      leaves[tc] = -1;
      continue;
    }
    int tOrig = treeOriginForest[tc];
    int *preds = predForest + tOrig;
    double *splitVal = splitForest + tOrig;
    double *scores = scoreForest + tOrig;
    Bump *bumps = bumpForest + tOrig;
    int *fs = facSplitForest + facOffForest[tc];

    int idx = 0;
    int pred = preds[idx];
    while (pred != leafPred) {
      //cout << "Pred "  << pred << " idx:  " << idx << endl;
      /*      if (pred < 0) {
	cout << splitVal[idx] << ":  " << fs[int(splitVal[idx])] << " / " << rowFT[-(pred+1)] << ", " << fs[int(splitVal[idx]) + rowFT[-(pred+1)]] << endl;
	for (int j = 0; j < Predictor::FacWidth(-(pred+1)); j++)
	  cout << fs[int(splitVal[idx]) + j];
	cout << endl;
	}*/
      Bump bump = bumps[idx];
      idx += (pred >= 0 ? (rowNT[pred] <= splitVal[idx] ?  bump.left : bump.right) : (fs[int(splitVal[idx]) + rowFT[-(pred+1)]] ? bump.left : bump.right));
      pred = preds[idx];
    }
    leaves[tc] = idx;
  }
  // TODO:  Instead of runtime check, can guarantee this by checking last level for non-negative
  // predictor fields.
}

// Each call writes one row.
//
void DecTree::QuantileLeaves(double *qRow, const int qCells, /*const int row,*/ const int leaves[]) {
  int *sampRanks = new int[qYLenForest];
  for (int i = 0; i < qYLenForest; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  int totRanks = 0;
  int tn;
  for (tn = 0; tn < nTree; tn++) {
    int predLeaf = leaves[tn];
    if (predLeaf < 0) // in-bag:  no OOB prediction at row for this tree.
      continue;
    int leafPos = qLeafPosForest[treeOriginForest[tn] + predLeaf];
    int leafExtent = qLeafExtentForest[treeOriginForest[tn] + predLeaf];
    int leafOff = qRankOriginForest[tn] + leafPos;
    for (int i = 0; i < leafExtent; i++) {
      int leafRank = qRankForest[leafOff+i];
      int rankCount = qRankCountForest[leafOff + i];
      //	cout << "Invalid rank:  "   << leafRank << " row  " << row << "  tree " << tn << ", absolute leaf off: " << treeOriginForest[tn] + predLeaf + i << ", abs. rank off:  "<<  leafOff +i << endl;
      sampRanks[leafRank] += rankCount;
      totRanks += rankCount;
    }
  }

  //  int qCells = QuantSig::qCells;
  //  double *qRow = QuantSig::qPred + row * qCells;
  double *countThreshold = new double[qCells];
  for (int i = 0; i < qCells; i++) {
    countThreshold[i] = totRanks * QuantSig::qVec[i];  // Rounding properties?
  }

  int qIdx = 0;
  int ranksSeen = 0;
  for (int i = 0; i < qYLenForest && qIdx < qCells; i++) {
     ranksSeen += sampRanks[i];
     while (qIdx < qCells && ranksSeen >= countThreshold[qIdx]) {
       //       cout << qIdx << ": " << ranksSeen << " / " << totRanks  << " , " << i << endl;
       qRow[qIdx++] = qYRankedForest[i];
     }
  }

  delete [] sampRanks;
  delete [] countThreshold;
}
