// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitsig.h

   @brief Class definitions for split signatures, which transmit splitting results to the index-tree splitting methods.

   @author Mark Seligman
 */

#ifndef ARBORIST_SPLITSIG_H
#define ARBORIST_SPLITSIG_H

#include "predictor.h"

/**
 @brief Split signatures transfer splitting information to inter-level mediation.
*/

class SplitSig {
  static int nPred;
  static SplitSig *levelSS; // The SplitSig records available for the current level.
  // Returns pointer to SplitSig in the level workspace.  SplitSigs are stored
  // with predictors as the fastest-varying index.  If no predictor index is passed,
  // then the vector based at 'splitIdx' is returned.
  //
 protected:
  /**
     @brief Looks up the SplitSig associated with a given pair.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return pointer to looked-up SplitSig.
   */
  static inline SplitSig *Lookup(int splitIdx, int predIdx = 0) {
    return levelSS + splitIdx * nPred + predIdx;
  }

 public:
  int level;  // Most recent level at which this record was stamped.
  int predIdx; // Helpful, but necessary, for example, if reusing records.
  int sCount; // # samples subsumed by split LHS.
  int lhIdxCount; // Index count of split LHS.
  double info; // Information content of split.

  // 'fac' information guides unpacking of LH the bit set for factor-valued
  // splitting predictors.
  //
  // Ideally, there would be SplitSigFac and SplitSigNum subclasses, with
  // Replay() and NonTerminal() methods implemented virtually.  Coprocessors
  // do not support vitual invocation, however, so we opt for a less
  // elegant solution.
  //
  struct {
      int bitOff; // Starting bit offset in pretree.
      int lhTop; // Reference for unpacking.
  } fac;

  static void TreeInit(int _levelMax);
  static void Factory(int _levelMax, int _nPred);
  static void ReFactory(int _levelMax);
  static void DeFactory();
  static SplitSig* ArgMax(int splitIdx, int _level, double preBias, double minInfo);
  double MinInfo();

  void LHSizes(int &_lhSCount, int &_lhIdxCount) {
    _lhSCount = sCount;
    _lhIdxCount = lhIdxCount;
  }

  double Replay(int splitIdx, int ptL, int ptR);

  /**
     @brief Sets splitting fields for a numerical predictor.

     @param splitIdx is the index node index.

     @param _predIdx is the predictor index.

     @param _level is the current level.

     @param _sCount is the count of samples in the LHS.

     @param _lhIdxCount is count of indices associated with the LHS.

     @param _info is the splitting information value, currently Gini.

     @return void.
   */
  static inline void WriteNum(int splitIdx, int _predIdx, int _level, int _sCount, int _lhIdxCount, double _info) {
    SplitSig *ssn = Lookup(splitIdx, _predIdx);
    ssn->level = _level;
    ssn->sCount = _sCount;
    ssn->lhIdxCount = _lhIdxCount;
    ssn->info = _info;
  }
  void NonTerminalNum(int level, int lhStart, int ptId);

  // Count and ordinal offset only needed until lowering, at which time the bits
  // for the LHS are set in a temporary vector for the level at hand.
  //
  void NonTerminalFac(int splitIdx, int ptId);

  /**
     @brief Sets splitting fields for a factor-value predictor.

     @param splitIdx is the index node index.

     @param _predIdx is the predictor index.

     @param _level is the current level.

     @param _sCount is the count of samples in the LHS.

     @param _lhIdxCount is count of indices associated with the LHS.

     @param _info is the splitting information value, currently Gini.

     @param _lhTop is the index of the highest sorted LHS rank.

     @return void.
   */
  static inline void WriteFac(int splitIdx, int _predIdx, int _level, int _sCount, int _lhIdxCount, double _info, int _lhTop) {
    SplitSig *ssf = Lookup(splitIdx, _predIdx);
    ssf->level = _level;
    ssf->sCount = _sCount;
    ssf->lhIdxCount = _lhIdxCount;
    ssf->info = _info;
    ssf->fac.lhTop = _lhTop;
  }

  /**
     @brief Dispatches nonterminal method based on predictor type.

     @param splitIdx is the index node index.

     @param level is the current level.

     @param ptId is the pretree index.

     @param lhStart is the start index of the LHS.

     @return void.

     Sacrifices elegance for efficiency, as virtual calls are not supported on coprocessor.
  */
  inline void NonTerminal(int splitIdx, int level, int ptId, int lhStart) {
    if (Predictor::FacIdx(predIdx) >= 0)
      NonTerminalFac(splitIdx, ptId);
    else
      NonTerminalNum(level, lhStart, ptId);
  }
};

#endif
