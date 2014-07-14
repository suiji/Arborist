/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_LEVEL_H
#define ARBORIST_LEVEL_H
#include <cfloat>

// Contains the sample data used by predictor-specific sample-walking pass.
// SampleOrds appear in predictor order, grouped by node.  They store the
// y-value, run class and sample index for the predictor position to which they
// correspond.
//
class SampleOrd {
 public:
  double yVal; // sum of response values associated with sample.
  int rowRun; // # occurrences of row sampled.  << # rows.
  int rank; // True rank, with ties identically receiving lowest applicable value.
  int sampleIdx; // RV index for this row.  Used by CTG as well as on replay.
};

class SampleOrdCtg : public SampleOrd {
 public:
  int ctg; // Response value:  zero-based.
};

// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class Level {
  static int nSamp; // Used only for SampleOrd coordination.
  static int *targIdx;
  static void Factory(int _nPred, int _nSamp, int _stCount);//int maxWidth);
  static void DeFactory();
 protected:
  static inline int SampleOff(int predIdx, int level) {
    return nSamp * (2*predIdx + ((level & 1) > 0 ? 1 : 0));
  }
  static int stCount;
  static Level *node;
 public:
  static void ReFactory(int _stCount);
  static class SplitSigNum *levelSSNum;
  static class SplitSigFac *levelSSFac;
  static int bagCount;
  static void FactoryReg(int maxWidth);
  static void ReFactoryReg(int _stCount);
  static int FactoryCtg(int maxWidth, int _ctgWidth);
  static void ReFactoryCtg(int _stCount);
  static void DeFactoryReg();
  static void DeFactoryCtg();
  static void TreeInit(double sum, int _bagCount);
};

class LevelReg : public Level {
 protected:
  static SampleOrd *sampleOrdRegWS;
 public:
  void LevelZero(int predIdx);
};

class LevelRegNum : public LevelReg {
 public:
  static void Split(int predIdx, const Node *node, int liveCount, int level);
  static void Factory();
  static void ReFactory();
  static void DeFactory();
};

class LevelRegFac : public LevelReg {
 public:
  static void Split(int predIdx, const Node *node, int liveCount, int level);
  static void Factory();
  static void ReFactory();
  static void DeFactory();
};

class LevelCtg : public Level {
 protected:
  SampleOrdCtg *sampleOrdCtgWS;
 public:
  void LevelZero(int predIdx);
};

class LevelCtgNum : public LevelCtg {
  // Gini coefficient is non-negative:  quotient of non-negative quantities.
  //  static const double giniMin = -1.0e25;

  // Mininum denominator value at which to test a split
  static const double minDenom = 1.0e-5;
  static int ctgWidth;
 public:
  static double *ctgSumR;
  static void Factory(const int _ctgWidth);
  static void ReFactory();
  static void DeFactory();
  static void Split(int predIdx, const NodeCtg *node, int liveCount, int level);
};

class LevelCtgFac : public LevelCtg {
  static int ctgWidth;
  static const int maxWidthDirect = 10; // Maximum width for which direct sampling used.
  static int *wideOffset; // Offsets into sample array for wide factors.
  static void SetWide(); // Sets the offsets once per simulation.
  static int totalWide; // Sum of wide offsets.
  static double *rvWide; // Random variates for wide factors.
 public:
  static int Factory(int _ctgWidth);
  static void ReFactory();
  static void DeFactory();
  static void TreeInit(double *_rvWide);
  static void ClearTree();
  static void Split(int predIdx, const NodeCtg *node, int liveCount, int level);
};
#endif
