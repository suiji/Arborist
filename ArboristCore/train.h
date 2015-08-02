// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file train.h

   @brief Class definitions for the training entry point.

   @author Mark Seligman
 */

#ifndef ARBORIST_TRAIN_H
#define ARBORIST_TRAIN_H

/**
   @brief Interface class for front end. Holds simulation-specific parameters of the data.
*/
class Train {
  static int treeBlock; // Multinode blocking parameter.
  static int nRow;
 protected:
  static int TrainZero(const class PredOrd *predOrd);
  static int TrainBlock(const class PredOrd *predOrd, int tn, int count);
  static int TrainForest(const class PredOrd *predOrd, int treeCount);
 public:
  static int nTree;
  static double probCutoff;
  static void ResponseReg(double y[]);
  static int ResponseCtg(const int y[], double yPerturb[], const double l[], int ln);
  static int Training(int minH, bool _quantiles, double minRatio, int totLevels, int &facWidth, int &totBagCount);
  static void Immutables(int _nTree, int _treeBlock, int _nRow);
  static void DeImmutables();

  static void SampleWeights(double sWeight[]);
  static void WriteForest(int *rPreds, double *rSplits, int *rBump, int* rOrigins, int *rFacOff, int *rFacSplits);
  static void WriteQuantile(double rQYRanked[], int rQRank[], int rQSCount[]);
};
#endif
