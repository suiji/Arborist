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

#include "typeparam.h"

/**
   @brief Records left-hand split specification derived by splitting method.
   The right-hand characteristics can be derived from the parent IndexSet
   and left-hand specification.
 */
class NuxLH {
  double info; // Information content of split.
  unsigned int idxStart; // Not derivable from index node alone.
  unsigned int lhExtent; // Index count of split LHS.
  unsigned int sCount; // # samples subsumed by split LHS.
  RankRange rankRange; // Numeric only.
  unsigned int lhImplicit; // Numeric only.
 public:


  /**
     @brief Records specifications derived by splitting method.

     @param idxStart is the starting LH SamplePred offset.

     @param lhExtent is the LH SamplePred extent.

     @param sCount is the number of samples subsumed by the LH.

     @param info is the information content inducing the split.

     @return void.
   */
  void inline Init(unsigned int idxStart,
		   unsigned int lhExtent,
		   unsigned int sCount,
		   double info) {
    this->idxStart = idxStart;
    this->lhExtent = lhExtent;
    this->sCount = sCount;
    this->info = info;
    // TODO:  'noRank' i/o zero:
    this->rankRange.rankLow = this->rankRange.rankHigh = 0;
  }

  
  /**
     @brief Bulk setter method for splits associated with numeric predictor.
     Passes through to generic Init(), with additional rank and implicit-count
     initialization.

     With introduction of dense ranks, splitting ranks can no longer be
     inferred by position alone.  Hence ranks are passed explicitly.

     @return void.
  */
  void inline InitNum(unsigned int idxStart,
		      unsigned int lhExtent,
		      unsigned int sCount,
		      double info,
		      unsigned int rankLow,
		      unsigned int rankHigh,
		      unsigned int lhImplicit = 0) {
    Init(idxStart, lhExtent, sCount, info);
    this->rankRange.rankLow = rankLow;
    this->rankRange.rankHigh = rankHigh;
    this->lhImplicit = lhImplicit;
  }


  /**
     @brief Bulk getter method.
   */
  void Ref(unsigned int &_idxStart,
	   unsigned int &_lhExtent,
	   unsigned int &_sCount,
	   double &_info,
	   RankRange &_rankRange,
	   unsigned int &_lhImplicit) const {
    _idxStart = this->idxStart;
    _lhExtent = this->lhExtent;
    _sCount = this->sCount;
    _info = this->info;
    _rankRange = this->rankRange;
    _lhImplicit = this->lhImplicit;
  }
};


/**
   @brief SSNode records sample, index and information content for a
   potential split at a given split/predictor pair.

  Ideally, there would be SSNodeFac and SSNodeNum subclasses, with
  Replay() and NonTerminal() methods implemented virtually.  Coprocessor
  may not support virtual invocation, however, so we opt for a less
  elegant solution.
 */
class SSNode {

  /**
     @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

     @param index is the current level's node environment.

     @param iSet is the node being split.

     @param preTree is the crescent pretree.
  
     @param run specifies the run sets associated with the node.

     @return true iff LH is implicit.
  */
  bool BranchRun(class IndexLevel *index,
		 class IndexSet *iSet,
		 class PreTree *preTree,
		 class Run *run) const;

  
  /**
     @brief Distributes LH/RH specification precipitated by a factor-valued
     splitting predictor.

     With LH and RH PreTree indices known, the sample indices associated with
     this split node can be looked up and remapped.  Replay() assigns actual
     index values, irrespective of whether the pre-tree nodes at these indices
     are terminal or non-terminal.

     @param index is the current level's node environment.

     @param iSet is the node being split.

     @param preTree is the crescent pretree.
  
     @param run specifies the run sets associated with the node.

     @return void.
 */
  void ReplayRun(class IndexLevel *index,
		 class IndexSet *iSet,
		 class PreTree *preTree,
		 const class Run *run) const;


  /**
     @brief Writes PreTree nonterminal node for numerical predictor.

     @param index is the current level's index environment.

     @param iSet is the index node being split.

     @param preTree is the crescent pre-tree.

     @return true iff LH is explicit.
  */
  bool BranchNum(class IndexLevel *index,
		 class IndexSet *iSet,
		 class PreTree *preTree) const;

  
  /**
     @brief Distributes LH/RH specification precipitated by a numerical
     splitting predictor.

     @param index is the Index environment for the current level.

     @param iSet is the node being split.

     @return void.
  */
  void ReplayNum(class IndexLevel *index,
		 class IndexSet *iSet) const;

  
 public:
  SSNode();
  double info; // Information content of split.
  unsigned int setIdx; // Index into RunSet workspace.
  unsigned int predIdx; // Rederivable, but convenient to cache.
  unsigned int sCount; // # samples subsumed by split LHS.
  unsigned int idxStart; // Dense packing causes value to vary.
  unsigned int lhExtent; // Index count of split LHS.
  RankRange rankRange; // Numeric only.
  unsigned int lhImplicit; // LHS implicit index count:  numeric only.
  unsigned char bufIdx; // Which of two buffers.

  static double minRatio; // Value below which never to split.

  /**
     @brief Pass-through from SplitPred.  Updates members to specifics of
     most informative split, if any.

     @param splitSig is the containing environment.

     @param splitIdx is the level-relative node index.

     @return void.
  */
  void ArgMax(const class SplitSig *splitSig,
	      unsigned int splitIdx);
  

  /**
     @brief Reports whether split is informative with respect to a threshold.

     @param minInfo outputs an information threhsold.

     @param sCount outputs the number of samples in LHS.

     @param lhExtent outputs the number of indices in LHS.

     @return true iff information content exceeds the threshold.
   */
  bool Informative(double &minInfo,
		   unsigned int &sCount,
		   unsigned int &lhExtent) const {
    if (info > minInfo) {
      minInfo = minRatio * info;
      sCount = this->sCount;
      lhExtent = this->lhExtent;
      return true;
    }
    else {
      return false;
    }
  }
  

  /**
     @brief Max reduction on the information content.

     @param gainMax outputs the running maximal information gain.

     @return true iff gainMax updated by the value passed.
   */
  double inline GainMax(double &gainMax) const {
    if (info > gainMax) {
      gainMax = info;
      return true;
    }
    else {
      return false;
    }
  }
  

  /**
     @brief Setter for the information value.

     @param 'info' is the information value to se.

     @return void, with side-effected 'info' field.
   */
  void inline SetInfo(double info) {
    this->info = info;
  }


  /**
     @brief Absorbs contents of an SSNode found to be arg-max.

     @param argMax is the arg-max node.

     @return void.
   */
  inline void Update(const SSNode *argMax) {
    if (argMax != nullptr) {
      *this = *argMax;
    }
  }
  

  /**
     @brief Dispatches nonterminal method based on predictor type.

     @param index is the current level's node environment.

     @param preTree is the crescent pretree.
  
     @param iSet is the node being split.

     @param run specifies the run sets associated with the node.

     @return true iff left-hand of split is explicit.
  */
  bool NonTerminal(class IndexLevel *index,
		   class PreTree *preTree,
		   class IndexSet *iSet,
		   class Run *run) const;
};


/**
  @brief Manages the SSNodes pertaining to a single level.
*/
class SplitSig {
  const unsigned int nPred;
  unsigned int splitCount;
  SSNode *levelSS; // Workspace records for the current level.

  
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
  inline SSNode &Lookup(unsigned int splitIdx,
			unsigned int predIdx = 0) {
    return levelSS[predIdx * splitCount + splitIdx];
  }


 public:

 SplitSig(unsigned int _nPred) :
  nPred(_nPred),
    splitCount(0),
    levelSS(nullptr) {
  }


  /**
     @brief Sets immutable static values.

     @param minRatio is an inf information content for splitting.  Must
     be non-negative, as otherwise ArgMax cannot distinguish splitting
     candidates from unset SSNodes, which have initial 'info' == 0.

     @return void.
  */
  static void Immutables(double minRatio);


  /**
     @brief Restores immutable state to default values.
  */  
  static void DeImmutables();


  /**
     @brief Walks predictors associated with a given split index to find which,
     if any, maximizes information gain above split's threshold.

     @param levelIdx is the level-relative index of the node.

     @param gainMax is the least information gain sufficient to splt the node.

     @return split record containing arg-max, if any.
  */
  SSNode *ArgMax(unsigned int levelIdx, double gainMax) const;

  
  /**
     @brief Allocates split signatures for a level.

     @param splitCount is the number of splitable nodes in the current level.

     @return void.
  */
  void LevelInit(unsigned int splitCount);


  /**
     @brief Deallocates level's signatures.

     @return void.
  */
  void LevelClear();


  /**
     @brief Setter for all splitting fields.

     @param splitIdx is the level-relative index of the node.

     @param predIdx is the index of the splitting predictor.

     @param setIdx is the index into the run-set workspace.

     @param bufIdx is the buffer index.

     @param nux records the specification derived by the splitting method.

     @return void.
  */
  void Write(unsigned int splitIdx,
	     unsigned int predIdx,
	     unsigned int setIdx,
	     unsigned int bufIdx,
	     const NuxLH &nux);
};

#endif
