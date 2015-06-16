// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file quant.h

   @brief Data structures and methods for predicting and writing quantiles.

   @author Mark Seligman

 */

#ifndef ARBORIST_QUANT_H
#define ARBORIST_QUANT_H
/**
 @brief Quantile signature.
*/
class Quant {
  static unsigned int smudge;
  static unsigned int logSmudge;
  static unsigned int binSize;
  static unsigned int qBin;
  static int nTree;
  static unsigned int nRow;
  static bool live;
  static int qCount;
  static double *qVec;
  static double *qPred;

  static int *treeBagCount;
  // Per-tree quantile vectors.
  static unsigned int **treeQRank;
  static int **treeQSCount;

  static unsigned int totBagCount; // Internally-maintained copy.
  static double *qYRankedForest;
  static int *qRankForest;
  static int *qSCountForest;
  static void Leaves(const int treeOriginForest[], const int predForest[], const int posForest[], const int leaves[], double qRow[]);
  static void SmudgeLeaves(const int treeOriginForest[], const int nonTermForest[], const int extentForest[], const int posForest[], int forestLength);
  static int RanksExact(int leafExtent, int rankOff, int sampRanks[]);
  static int RanksSmudge(unsigned int leafExtent, int rankOff, int sampRanks[]);
  static void Quantiles(const class PreTree *preTree, const int bump[], const int leafExtent[], unsigned int qRank[], int qSCount[]);
  static int *LeafPos(int treeHeight, const int bump[], const int leafExtent[]);
  static void AbsOffset(const int nonTerm[], const int leafExtent[], int treeSize, int absOff[]);
 public:
  static void FactoryTrain(unsigned int _nRow, int _nTree, bool _train);
  static void FactoryPredict(int _nTree, double qYRanked[], int qRank[], int qSCount[]);
  static void EntryPredict(double _qVec[], int _qCount, unsigned int _qBin, double _qPred[], unsigned int _nRow = 0);
  static void DeFactoryPredict();
  static void ConsumeTrees();
  static void TreeRanks(const class PreTree *pt, const int bump[], const int leafExtent[], int tn);
  static void Write(double rQYRanked[], int rQRank[], int rQSCount[]);
  static void PredictRows(const int treeOriginForest[], const int nonTermForest[], const int extentForest[], int forestLength, int predictLeaves[]);
};

#endif
