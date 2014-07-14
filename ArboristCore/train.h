/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_TRAIN_H
#define ARBORIST_TRAIN_H

// Interface class for front end.
// Holds simulation-specific parameters of the data.
//
class Train {
  static void PredWeight(const double weights[], const double predProb);
  static int SampleDraw(double draw);
  static int *SampleWith(int ln);
  static int *SampleWithout(const int sampVec[], int sourceLen, int targLeon);
public:
  static int nTree;
  static int nSamp;
  static double probCutoff;
  static bool doQuantiles;
  static bool sampReplace;
  static int qCells;
  static double *qVec;
  static double minRatio; // Spread between parent and child Gini gain.
  static int blockSize;
  static int accumRealloc;
  static int probResize;
  static double *sCDF;
  static int *cdfOff;
  static void IntBlock(int xBlock[], int _nrow, int _ncol);
  static void ResponseReg(double y[]);
  static int ResponseCtg(const int y[], double yPerturb[]);
  static void TrainInit(double *_predWeight, double _predProb, int _nTree, int _nSamp, bool _smpReplace, bool _doQuantile, double _minRatio, int _blockSize);
  static int Training(int minH, int *facWidth, int *totBagCount, int *totQLeafWidth, int totLevels);
  static void Factory(int _nTree, int _nSamp, bool _sampReplace, bool _quantiles, double _minRatio, int _blockSize);
  static void DeFactory();
  static void SampleWeights(double sWeight[]);
  static void WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBumpL, int *rBumpR, int* rOrigins, int *rFacOff, int * rFacSplits);
  static void WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]);
  static void Quantiles(double *_qVec, const int _qCells);

  // TODO:  Advance 'ruBase' sufficiently far to hold all liveCount * nPred
  // slots for current level.
  static inline bool Splitable(int predIdx, int nodeIdx) {
    return ruBase[nodeIdx] < Predictor::predProb[predIdx];
  }
};
#endif
