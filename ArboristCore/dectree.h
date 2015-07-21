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


/**
   @brief The decision forest is a collection of decision trees.  DecTree members and methods are currently all static.
*/
class DecTree {
  static int nTree;
  static unsigned int nRow; // Set separately for training and prediction.
  static int nPred;
  static int nPredNum;
  static int nPredFac;
  static int *treeSizes;
  static int *treeOriginForest;
  static int **predTree;
  static double **splitTree;
  static int **bumpTree;
  static int *treeFacWidth; // Per-tree:  # factors subsumed by splits.
  static int **treeFacSplits; // Per-tree:  temporary vectors holding factor values.
  static int *facOffForest;
  static int *facSplitForest; // Consolidation of per-tree values.

  static double *predInfo; // E.g., Gini gain.  May belong elsewhere, as known before scoring.
  static int* predForest;  // Split predictor / sample extent : nonterminal / terminal.
  static double* numForest; // Split value / score : nonterminal / terminal.
  static int* bumpForest;  // Successor offset / zero :  nonterminal / terminal.
  static unsigned int *inBag; // Train only.
  static int forestSize;
  static void ObsDeImmutables();

  static void ConsumeSplitBits(class PreTree *pt, int &treeFW, int *&treeFS);
  static void SetBagRow(const unsigned int inBag[], int treeNum);
  static bool InBag(int treeNum, unsigned int row);
  static void PredictAcrossCtg(int *census, unsigned int ctgWidth, bool useBag);
  static void Vote(const int *census, int yCtg[], int *confusion, double error[], unsigned int ctgWidth, bool useBag);
  static void PredictRowNumReg(unsigned int row, double[], int leaves[], bool useBag);
  static void PredictRowFacReg(unsigned int row, int rowT[], int leaves[], bool useBag);
  static void PredictRowMixedReg(unsigned int row, double rowNT[], int rowFT[], int leaves[], bool useBag);
  static void PredictRowNumCtg(unsigned int row, double rowT[], int rowPred[], bool useBag);
  static void PredictRowFacCtg(unsigned int row, int rowT[], int rowPred[], bool useBag);
  static void PredictRowMixedCtg(unsigned int row, double rowNT[], int rowIT[], int rowPred[], bool useBag);
  static void PredictAcrossNumReg(double prediction[], int *predictLeaves, bool useBag);
  static void PredictAcrossFacReg(double prediction[], int *predictLeaves, bool useBag);
  static void PredictAcrossMixedReg(double prediction[], int *predictLeaves, bool useBag);
  static void PredictAcrossNumCtg(int *census, unsigned int ctgWidth, bool useBag);
  static void PredictAcrossFacCtg(int *census, unsigned int ctgWidth, bool useBag);
  static void PredictAcrossMixedCtg(int *census, unsigned int ctgWidth, bool useBag);
  static void DeFactoryTrain();

  /**
     @brief computes the offset and bit coordinates of a given <tree, row> pair in the InBag structure.

     @param treeNum is the current tree index.

     @param row is the row index.

     @param off outputs the offset coordinate.

     @param bit outputs the bit coordinate.

     @return value of InBag slot.
   */
  static unsigned int inline BagCoord(int treeNum, unsigned int row, unsigned int &off, unsigned int &bit) {
    const unsigned int slotBits = 8 * sizeof(unsigned int);
    unsigned int idx = row * nTree + treeNum;
    off = idx / slotBits; // Compiler should generate right-shift.
    bit = idx & (slotBits - 1);
    return inBag[off];
  }
 public:
  static void ObsImmutables(int _nRow, int _nPred, int _nPredNum, int _nPredFac);
  static void DeFactoryPredict();
  static int BlockConsume(class PreTree *ptBlock[], int treeBlock, int treeStart);
  static void FactoryTrain(int _nTree);
  static int ConsumeTrees(int &cumFacWidth);
  static void ForestReload(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplits[]);
  static void ScaleInfo(double*);
  static void WriteForest(int *rPreds, double *rSplits, int *rBump, int* rOrigins, int *rFacOff, int * rFacSplits);
  static  void WriteTree(int treeNum, int tOrig, int treeFacOffset, int *outPreds, double* outSplitVals, int *outBump, int *outFacSplits);
  static void PredictCtg(int *census, unsigned int ctgWidth, int y[], int *confusion, double error[], bool useBag = true);
  static void PredictAcrossReg(double outVec[], bool useBag);
};

#endif
