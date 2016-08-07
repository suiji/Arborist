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

/**
   @brief SSNode records sample, index and information content for a
   potential split at a given split/predictor pair.

 */
class SSNode {  
  double NonTerminalRun(class SamplePred *samplePred, class PreTree *preTree, unsigned int splitIdx, int start, int end, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH, class Run *run);
  double NonTerminalNum(class SamplePred *samplePred, class PreTree *preTree, unsigned int splitIdx, int start, int end, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH);
 public:
  SSNode();
  int setIdx; // Index into RunSet workspace.
  unsigned int predIdx; // Rederivable, but convenient to cache.
  unsigned int sCount; // # samples subsumed by split LHS.
  unsigned int lhIdxCount; // Index count of split LHS.
  double info; // Information content of split.
  unsigned char bufIdx;
  
  static double minRatio;
  
  // Ideally, there would be SplitSigFac and SplitSigNum subclasses, with
  // Replay() and NonTerminal() methods implemented virtually.  Coprocessor
  // may not support virtual invocation, however, so we opt for a less
  // elegant solution.

  /**
   @brief Derives an information threshold.

   @return information threshold
  */
  double inline MinInfo() {
    return minRatio * info;
  }

  
  /**
     @brief Accessor for bipartitioning.

     @param _lhSCount outputs the number of samples in LHS.

     @param _lhIdxCount outputs the number of indices in LHS.

     @return void, with output reference parameters.
   */  
  void inline LHSizes(unsigned int &_lhSCount, unsigned int &_lhIdxCount) const {
    _lhSCount = sCount;
    _lhIdxCount = lhIdxCount;
  }

  
  double NonTerminal(class SamplePred *samplePred, class PreTree *preTree, unsigned int splitIdx, int start, int end, unsigned int ptId, unsigned int &ptL, unsigned int &ptR, class Run *run);
};


/**
  @brief SplitSigs manage the SSNodes for a given level instantation.
*/
class SplitSig {
  int splitCount;
  SSNode *levelSS; // Workspace records for the current level.
 protected:
  static unsigned int nPred;

  /**
     @brief Looks up the SplitSig associated with a given pair.

     SplitSigs are stored with split number as the fastest-varying
     index.  The likelihood of false sharing during splitting is
     fairly low, given that predictor selection is probabalistic
     and splitting workloads themselves are nonuniform.  Nonetheless,
     predictor-specific references are kept fairly far apart.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return pointer to looked-up SplitSig.
   */
  inline SSNode &Lookup(unsigned int splitIdx, unsigned int predIdx = 0) {
    return levelSS[predIdx * splitCount + splitIdx];
  }

 public:
  SSNode *ArgMax(unsigned int splitIdx, double minInfo) const;
  static void Immutables(unsigned int _nPred, double _minRatio);
  static void DeImmutables();

  void LevelInit(int splitCount);
  void LevelClear();
  void Write(unsigned int _splitIdx, unsigned int _predIdx, int _runIdx, unsigned int _bufIdx, unsigned int _sCount, unsigned int _lhIdxCount, double _info);
};

#endif
