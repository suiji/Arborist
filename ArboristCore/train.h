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
  static const int accumExp = 5;
  static int reLevel; // Diagnostic:  # reallocations.
  static int levelMax;
public:
  static int nTree;
  static int nSamp;
  static double probCutoff;
  static double minRatio; // Spread between parent and child information content.
  static int blockSize;
  static int accumRealloc;
  static int probResize;
  static double *sCDF;
  static int *cdfOff;
  static void ResponseReg(double y[]);
  static int ResponseCtg(const int y[], double yPerturb[], const double l[], int ln);
  static int Training(int minH, bool _quantiles, int totLevels, int &facWidth, int &totBagCount);
  static int Factory(int _nTree, double _minRatio, int _blockSize);
  static void DeFactory();
  static int ReFactory();
  static inline double MinInfo(double info) {
    return minRatio * info;
  }

  static void SampleWeights(double sWeight[]);
  static void WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBump, int* rOrigins, int *rFacOff, int * rFacSplits);
  static void WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]);
};
#endif
