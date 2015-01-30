// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file dectree.h

   @brief Data structures and methods for constructing and walking
       the decision tree.

   @author Mark Seligman

 */

#ifndef ARBORIST_DECTREE_H
#define ARBORIST_DECTREE_H

#include <climits>


/**
   @brief The decision forest is a collection of decision trees.  DecTree members and methods are currently all static.
*/
class DecTree {
  static int nTree; // Running tally of forest size.
  static int nRow;
  static int nPred;
  static int nPredNum;
  static int nPredFac;
  static int *treeSizes;
  static int *treeOriginForest;
  static int **treePreds;
  static double **treeSplits;
  static double **treeScores;
  static int **treeBumps;
  static int *treeFacWidth; // Per-tree:  # factors subsumed by splits.
  static int **treeFacSplits; // Per-tree:  temporary vectors holding factor values.
  static int *facOffForest;
  static int *facSplitForest; // Consolidation of per-tree values.

  static int *treeQRankWidth;
  static int *treeQLeafWidth;
  static int **treeQLeafPos;
  static int **treeQLeafExtent;
  static int **treeQRank;
  static int **treeQRankCount;
  static double *qYRankedForest;
  static int qYLenForest; // Length of training response.
  static int *qRankOriginForest;
  static int *qRankForest;
  static int *qRankCountForest;
  static int *qLeafPosForest;
  static int *qLeafExtentForest;

  static double *predGini; // 'tgini' == impurity.  May belong elsewhere, as known before scoring.
  static int* predForest;
  static double* splitForest;
  static double* scoreForest;
  static int* bumpForest;
  static unsigned int *inBag; // Train only.
  static int forestSize;

  // Only client for thse is quantile regression
  static int totBagCount;
  static int totQLeafWidth;

  static void ConsumeSplitBits(int treeNum, int facWidth);
  static void SetBagRow(const bool sampledRows[], int treeNum);
  static bool InBag(int treeNum, int row);
  static void PredictRowNumReg(int row, double[], int leaves[], bool useBag);
  static void PredictRowFacReg(int row, int rowT[], int leaves[], bool useBag);
  static void PredictRowMixedReg(int row, double rowNT[], int rowFT[], int leaves[], bool useBag);
  static void PredictRowNumCtg(int row, double rowSlice[], int ctgWidth, int rowPred[], bool useBag);
  static void PredictRowFacCtg(int row, int rowFT[], int ctgWidth, int rowPred[], bool useBag);
  static void PredictRowMixedCtg(int row, double rowNT[], int rowFT[], int ctgWidth, int rowPred[], bool useBag);
  static void PredictAcrossNumReg(double prediction[], bool useBag);
  static void PredictAcrossFacReg(double prediction[], bool useBag);
  static void PredictAcrossMixedReg(double prediction[], bool useBag);
  static void QuantileLeaves(double qRow[], int qCells, const int leaves[]);
  static void QuantileRanks(int treeNum, int treeSize, int bagCount);
  static void DeForest();
 public:
  static void DeForestPredict();
  static const int leafPred = INT_MIN; // Positive counterpart not representable as int.
  static void ConsumePretree(const bool _inBag[], int bagCount, int treeSize, int treeNum);
  static void FactoryTrain(int _nTree, int _nRow, int _nPred, int _nPredNum, int _nPredFac);
  static int AllTrees(int *cumFacWidth, int *cumBagWidth, int *totQLeafWidth);
  static void ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], double _scores[], int _bump[], int _origins[], int _facOff[], int _facSplits[]);
  static void ForestReloadQuant(double qYRanked[], int qYLen, int qRank[], int qRankOrigin[], int qRankCount[], int qLeafPos[], int qLeafExtent[]);
  static void ScaleGini(double*);
  static void WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBump, int* rOrigins, int *rFacOff, int * rFacSplits);
  static  void WriteTree(int treeNum, int tOrig, int treeFacOffset, int *outPreds, double* outSplitVals, double* outScores, int *outBump, int *outFacSplits);
  static void WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]);
  static void PredictAcrossReg(double outVec[], bool useBag=true);
  static void PredictAcrossCtg(int yCtg[], int ctgWidth, int confusion[], double error[], bool useBag = true);
};
#endif
