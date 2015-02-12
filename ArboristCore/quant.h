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
  static int nTree;
  static int nRow;
  static bool live;

  // Per-tree quantile vectors.
  static int *treeQRankWidth;
  static int **treeQLeafPos;
  static int **treeQLeafExtent;
  static int **treeQRank;
  static int **treeQRankCount;

  static int totBagCount; // Internally-maintained copy.
  static int forestSize; // " " 
  static double *qYRankedForest;
  static int qYLenForest; // Length of training response.
  static int *qRankOriginForest;
  static int *qRankForest;
  static int *qRankCountForest;
  static int *qLeafPosForest;
  static int *qLeafExtentForest;
  static void Leaves(const int treeOriginForest[], const int leaves[], double qRow[]);
 public:
  static void FactoryTrain(int _nRow, int _nTree, bool _train);
  static void FactoryPredict(int _nTree, double qYRanked[], int qYLen, int qRank[], int qRankOrigin[], int qRankCount[], int qLeafPos[], int qLeafExtent[]);
  static void EntryPredict(double _qVec[], int _qCount, double _qPred[], int _nRow = 0);
  static void DeFactoryPredict();
  static void ConsumeTrees(const int treeOriginForest[], int forestSize);
  static void TreeRanks(int tn, int treeSize, int bagCount);
  static void Write(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]);
  static void PredictRows(const int treeOriginForest[], int *predictLeaves);
  static int qCount;
  static double *qVec;
  static double *qPred;
};

#endif
