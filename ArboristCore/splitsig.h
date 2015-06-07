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
  double NonTerminalFac(class SamplePred *samplePred, class PreTree *preTree, class SplitPred *splitPred, int level, int start, int end, int ptLH, int ptRH, int facCard, double &splitVal);
  double NonTerminalNum(class SamplePred *samplePred, class PreTree *preTree, int level, int start, int end, int ptLH, int ptRH, double &splitVal);
 public:
 SSNode() : info(0.0) {}
  int predIdx; // Helpful, but necessary, for example, if reusing records.
  int splitIdx; // Helpful, but not necessary.
  int sCount; // # samples subsumed by split LHS.
  int lhIdxCount; // Index count of split LHS.
  double info; // Information content of split.

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
  void inline LHSizes(int &_lhSCount, int &_lhIdxCount) {
    _lhSCount = sCount;
    _lhIdxCount = lhIdxCount;
  }

  double NonTerminal(class SamplePred *samplePred, class PreTree *preTree, class SplitPred *splitPred, int level, int start, int end, int ptId, int &ptL, int &ptR);
};


/**
  @brief SplitSigs manage the SSNodes for a given level instantation.
*/
class SplitSig {
  int splitCount;
  SSNode *levelSS; // Workspace records for the current level.
 protected:
  static int nPred;

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
  inline SSNode &Lookup(int splitIdx, int predIdx = 0) {
    return levelSS[predIdx * splitCount + splitIdx];
  }

 public:
  SSNode *ArgMax(int splitIdx, double minInfo) const;
  static void Immutables(int _nPred, double _minRatio);
  static void DeImmutables();
  
  /**
     @brief Sets splitting fields for a splitting predictor.

     @param splitIdx is the index node index.

     @param _predIdx is the predictor index.

     @param _sCount is the count of samples in the LHS.

     @param _lhIdxCount is count of indices associated with the LHS.

     @param _info is the splitting information value, currently Gini.

     @return void.
   */
  inline void Write(int _splitIdx, int _predIdx, int _sCount, int _lhIdxCount, double _info) {
    SSNode ssn;
    ssn.splitIdx = _splitIdx;
    ssn.predIdx = _predIdx;
    ssn.sCount = _sCount;
    ssn.lhIdxCount = _lhIdxCount;
    ssn.info = _info;
    Lookup(_splitIdx, _predIdx) = ssn;
  }

  void LevelInit(int splitCount);
  void LevelClear();

};

#endif
