// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runset.h

   @brief Definitions for the Run classes, which maintain predictor
   runs, especially factor-valued predictors.

   @author Mark Seligman
 */

#ifndef SPLIT_RUNSET_H
#define SPLIT_RUNSET_H

#include <vector>

#include "pretree.h"
#include "sumcount.h"

/**
   FRNodes hold field values accumulated from runs of factors having the
   same value.

   That is, they group factor-valued predictors into block representations.
   These values live for a single level, so must be consumed before a new
   level is started.
 */
class FRNode {
 public:
  unsigned int rank; // Same 0-based value as internal code.
  IndexT sCount; // Sample count of factor run:  need not equal length.
  double sum; // Sum of responses associated with run.
  IndexRange range;

  FRNode() : sCount(0), sum(0.0), range(IndexRange()) {
  }

  bool isImplicit();


  /**
     @brief Initializer.
   */
  inline void init(unsigned int rank,
                   IndexT sCount,
                   double sum,
                   IndexT start,
                   IndexT extent) {
    this->rank = rank;
    this->sCount = sCount;
    this->sum = sum;
    this->range = IndexRange(start, extent);
  }

  
  /**
     @brief Range accessor.  N.B.:  Should not be invoked on dense
     run, as 'start' will hold a reserved value.

     @return range of indices subsumed by run.
   */
  inline IndexRange getRange() const {
    return range;
  }


  /**
     @brief Accumulates run contents into caller.
   */
  inline void accum(IndexT& sCount,
                    double& sum) const {
    sCount += this->sCount;
    sum += this->sum;
  }

  /**
     @brief Rank getter.

     @return rank.
   */
  inline auto getRank() const {
    return rank;
  }
};


/**
   @brief Ad hoc container for simple priority queue.
 */
struct BHPair {
  double key;  // Comparitor value.
  unsigned int slot; // Slot index.
};


/**
   RunSets live only during a single level, from argmax pass one (splitting)
   through argmax pass two.  They accumulate summary information for split/
   predictor pairs anticipated to have two or more distinct runs.  RunSets
   are not yet built for numerical predictors, which have so far been
   generally assumed to have dispersive values.

   The runCounts[] vector tracks conservatively-estimated run lengths for
   every split/predictor pair, regardless whether the pair is chosen for
   splitting in a given level (cf., 'mtry' and 'predProb').  The vector
   must be reallocated at each level, to accommodate changes in node numbering
   introduced through splitting.

   Run lengths for a given predictor decrease, although not necessarily
   monotonically, with splitting.  Hence once a pair becomes a singleton, the
   fact is worth preserving for the duration of training.  Numerical predictors
   are assigned a nonsensical run length of zero, which is changed to a sticky
   value of unity, should a singleton be identified.  Run lengths are
   transmitted between levels duing restaging, which is the only phase to
   maintain a map between split nodes and their descendants.  Similarly, new
   singletons are very easy to identify during restaging.

   Other than the "bottom" value of unity, run lengths can generally only be
   known precisely by first walking the predictor ranks.  Hence a conservative
   value is used for storage allocation, namely, that obtained during a previous
   level.  Note that this value may be quite conservative, as the pair may not
   have undergone a rank-walk in the previous level.  The one exception to this
   is the case of an argmax split, for which both left and right run counts are
   known from splitting.
*/
class RunSet {
  bool hasImplicit; // Whether dense run present.
  int runOff; // Temporary offset storage.
  int heapOff; //
  int outOff; //
  FRNode *runZero; // Base for this run set.
  BHPair *heapZero; // Heap workspace.
  unsigned int *outZero; // Final LH and/or output for heap-ordered slots.
  double *ctgZero; // Categorical:  run x ctg checkerboard.
  double *rvZero; // Non-binary wide runs:  random variates for sampling.
  unsigned int runCount;  // Current high watermark:  not subject to shrinking.
  unsigned int runsLH; // Count of LH runs.


  /**
     @brief Caches response sums.

     @param nodeSum is the per-category response over the node (IndexSet).
   */
  void setSumCtg(const vector<double> &nodeSum);


 public:
  static constexpr unsigned int maxWidth = 10; // Algorithmic threshold.
  static IndexT noStart; // Inattainable index.
  unsigned int safeRunCount;

  RunSet() : hasImplicit(false), runOff(0), heapOff(0), outOff(0), runZero(0), heapZero(0), outZero(0), ctgZero(0), rvZero(0), runCount(0), runsLH(0), safeRunCount(0) {}


  /**
     @brief Determines whether it is necessary to expose the right-hand runs.

     Right-hand runs can often be omitted from consideration by
     presetting a split's next-level contents all to the right-hand
     index, then overwriting those known to lie in the left split.  The
     left indices are always exposed, making this a convenient strategy.
   
     This cannot be done if the left contains an implicit run, as implicit
     run indices are not directly recorded.  In such cases a complementary
     strategy is employed, in which all indices are preset to the left
     index, with known right-hand indices overwritten.  Hence the
     right-hand runs must be enumerated in such instances.

     @return true iff right-hand runs must be exposed.
  */
  bool implicitLeft() const;

  /**
     @brief Builds a run for the dense rank using residual values.

     @param cand is the splitting candidate with potential implicit state.

     @param denseRank is the rank of the implicit blob, if any.

     @param ctgSum is the per-category response over the node (IndexSet).
  */
  void writeImplicit(const class SplitNux* cand,
                     const class SplitFrontier* sp,
                     const vector<double>& ctgSum = vector<double>(0));

  /**
     @brief Hammers the pair's run contents with runs selected for
     sampling.

     Since the runs are to be read numerous times, performance
     may benefit from this elimination of a level of indirection.

     @return post-shrink run count.
  */
  unsigned int deWide(PredictorT nCtg);

  /**
     @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.

     @param pop is the number of elements to pop from the heap.
  */
  void dePop(unsigned int pop = 0);

  /**
     @brief Updates local vector bases with their respective offsets,
     addresses, now known.
  */
  void reBase(vector<FRNode> &facRun,
              vector<BHPair> &bHeap,
              vector<unsigned int> &lhOut,
              vector<double> &ctgSum,
              PredictorT nCtg,
              vector<double> &rvWide);

  /**
     @brief Records only the (casted) relative vector offsets, as absolute
     base addresses not yet known.
  */
  void offsetCache(unsigned int _runOff, unsigned int _heapOff, unsigned int _outOff);

  /**
     @brief Writes to heap arbitrarily:  sampling w/o replacement.
  */
  void heapRandom();

  /**
     @brief Writes to heap, weighting by slot mean response.
  */
  void heapMean();

  /**
     @brief Writes to heap, weighting by category-1 probability.
  */
  void heapBinary();


  /**
     @brief Redirects samples to left or right according.

     @param cand is a splitting candidate.

     @param iSet encodes the sample indices associated with the split.

     @param preTree is the crescent pre-tree.

     @param index is the index set environment for the current level.

     @param[out] replayLeft outputs true iff split LHS contains implicit runs.

     @return sum of replayed response.
   */
  double branch(class IndexSet* iSet,
                class PreTree* preTree,
                const class SplitFrontier* splitFrontier,
		class Replay* replay,
                vector<SumCount>& ctgCrit,
                bool& replayLeft) const;


  /**
     @brief Subtracts a run's per-category responses from the current run.

     @param nCtg is the response cardinality.

     @param runIdx is the run index.
   */
  void residCtg(PredictorT nCtg,
                unsigned int runIdx);


  /**
     @brief Accessor for runCount field.

     @return reference to run count.
   */
  inline auto getRunCount() const {
    return runCount;
  }


  inline void setRunCount(unsigned int runCount) {
    this->runCount = runCount;
  }
  
  
  inline unsigned int getSafeCount() const {
    return safeRunCount;
  }
  
  
  /**
     @brief "Effective" run count, for the sake of splitting, is the lesser
     of the true run count and 'maxWidth'.

     @return effective run count
   */
  inline PredictorT effCount() const {
    return runCount > maxWidth ? maxWidth : runCount;
  }


  /**
     @brief Accumulates contents at position referenced by a given index.

     @param outPos is an index in the output vector.

     @param sCount[in, out] accumulates sample count using the output position.

     @param sum[in, out] accumulates the sum using the output position.
   */
  inline void sumAccum(unsigned int outPos, IndexT& sCount, double& sum) {
    unsigned int slot = outZero[outPos];
    runZero[slot].accum(sCount, sum);
  }

  /**
     @brief Sets run parameters and increments run count.
   */
  inline void write(PredictorT rank,
                    IndexT sCount,
                    double sum,
                    IndexT extent,
                    IndexT start = noStart) {
    runZero[runCount++].init(rank, sCount, sum, start, extent);
    hasImplicit = (start == noStart);
  }


  /**
     @brief Accumulates checkerboard values prior to writing topmost
     run.

     @return void.
   */
  inline void accumCtg(PredictorT nCtg,
                       double ySum,
                       PredictorT yCtg) {
    ctgZero[runCount * nCtg + yCtg] += ySum;
  }


  /**
     @return checkerboard value at slot for category.
   */
  inline double getSumCtg(PredictorT slot,
			  PredictorT nCtg,
			  PredictorT yCtg) const {
    return ctgZero[slot * nCtg + yCtg];
  }


  /**
     @brief Looks up the two binary response sums associated with a given
     output slot.

     @param outPos is a position in the output vector.

     @param[in, out] sum0 accumulates the response at rank 0.

     @param[in, out] sum1 accumulates the response at rank 1.

     @return true iff slot counts differ by at least unity.
   */
  inline bool accumBinary(unsigned int outPos,
			  double& sum0,
			  double& sum1) {
    unsigned int slot = outZero[outPos];
    double cell0 = getSumCtg(slot, 2, 0);
    sum0 += cell0;
    double cell1 = getSumCtg(slot, 2, 1);
    sum1 += cell1;

    IndexT sCount = runZero[slot].sCount;
    unsigned int slotNext = outZero[outPos+1];
    // Cannot test for floating point equality.  If sCount values are unequal,
    // then assumes the two slots are significantly different.  If identical,
    // then checks whether the response values are likely different, given
    // some jittering.
    // TODO:  replace constant with value obtained from class weighting.
    return sCount != runZero[slotNext].sCount ? true : getSumCtg(slotNext, 2, 1) - cell1 > 0.9;
  }


  /**
    @brief Outputs sample and index counts at a given slot.

    @param liveIdx is a cached offset for the pair.

    @param pos is the position to dereference in the rank vector.

    @param sCount outputs the sample count.

    @return total index count subsumed, with reference accumulator.
  */
  inline IndexT lHCounts(unsigned int slot, IndexT& sCount) const {
    FRNode *fRun = &runZero[slot];
    sCount = fRun->sCount;
    return  fRun->range.getExtent();
  }


  inline auto getRunsLH() const {
    return runsLH;
  }


  unsigned int getRank(unsigned int outSlot) const;

  /**
     @brief Decodes bit vector of slot indices and stores LH indices.

     @param lhBits encodes LH/RH slot indices as on/off bits, respectively.

     @param lhSampCt outputs the LHS sample count.

     @return LHS index count.
  */
  IndexT lHBits(unsigned int lhBits, IndexT& lhSampCt);
  
  /**
     @brief Dereferences out slots and accumulates splitting parameters.

     @param cut is the final out slot of the LHS:  < 0 iff no split.

     @param lhSampCt outputs the LHS sample count.

     @return LHS index count.
  */
  unsigned int lHSlots(unsigned int outPos, IndexT& lhSampCt);

  /**
     @brief Looks up run parameters by indirection through output vector.
     
     N.B.:  should not be called with a dense run.

     @return index range associated with run.
  */
  IndexRange getBounds(unsigned int outSlot) const;
};


/**
   @brief  Runs only:  caches pre-computed workspace starting indices to
   economize on address recomputation during splitting.

   Run objects are allocated per-tree, and live throughout training.
*/
class Run {
  unsigned int setCount;
  vector<RunSet> runSet;
  vector<FRNode> facRun; // Workspace for FRNodes used along level.
  vector<BHPair> bHeap;
  vector<unsigned int> lhOut; // Vector of lh-bound slot indices.
  vector<double> ctgSum;
  vector<double> rvWide;

  /**
     @brief Adjusts offset and run-count fields of each RunSet.
  */
  void reBase();

  void runSets(const vector<unsigned int>& safeCount);


public:
  const unsigned int ctgWidth;  // Response cardinality; zero iff numerical.

  /**
     @brief Constructor.

     @param ctgWidth_ is the response cardinality.

     @param nRow is the number of training rows:  inattainable offset.
  */
  Run(unsigned int ctgWidth_,
      unsigned int nRow);

  /**
     @brief Clears workspace used by current level.
   */
  void clear();

  /**
     @brief Regression:  all runs employ a heap.

     @param safeCount enumerates conservative run counts.
  */
  void offsetsReg(const vector<unsigned int> &safeCount);


  /**
     @brief Classification:  only wide run sets use the heap.

     @param safeCount as above.
  */
  void offsetsCtg(const vector<unsigned int> &safeCount);

  /**
     @brief Redirects samples to left or right according.

     @param cand is a splitting candidate.

     @param iSet encodes the sample indices associated with the split.

     @param preTree is the crescent pre-tree.

     @param index is the index set environment for the current level.

     @param[out] replay left outputs true iff split LHS contains implicit runs.

     @return sum of replayed response.
   */
  double branch(const class SplitFrontier* splitFrontier,
                class IndexSet* iSet,
                class PreTree* preTree,
		class Replay* replay,
                vector<SumCount>& ctgCrit,
                bool& replayLeft) const;

  
  /**
     @brief Accessor for RunSet at specified index.
   */
  inline RunSet *rSet(unsigned int rsIdx) {
    return &runSet[rsIdx];
  }

  
  /**
     @brief Presets runCount field to a conservative value for
     the purpose of allocating storage.

     @param idx specifies an internal index.

     @param count is the "safe" count value.
   */
  void setSafeCount(unsigned int idx, unsigned int count) {
    runSet[idx].safeRunCount = count;
  }

  
  /**
     @brief Gets safe count associated with a given index.
   */
  unsigned int getSafeCount(unsigned int idx) const {
    return runSet[idx].safeRunCount;
  }
};


/**
   @brief Implementation of binary heap tailored to RunSets.

   Not so much a class as a collection of static methods.
   TODO:  Templatize and move elsewhere.
*/
struct BHeap {
  /**
     @brief Determines index of parent.
   */
  static inline int parent(int idx) { 
    return (idx-1) >> 1;
  };


  /**
     @brief Empties the queue.

     @param pairVec are the queue records.

     @param[out] lhOut outputs the popped slots, in increasing order.

     @param pop is the number of elements to pop.  Caller enforces value > 0.
  */
  static void depopulate(BHPair pairVec[],
                         unsigned int lhOut[],
                         unsigned int pop);

  /**
     @brief Inserts a key, value pair into the heap at next vacant slot.

     Heap updates to move element with maximal key to the top.

     @param pairVec are the queue records.

     @param slot_ is the slot position.

     @param key_ is the associated key.
  */
  static void insert(BHPair pairVec[],
                     unsigned int slot_,
                     double key_);

  /**
     @brief Pops value at bottom of heap.

     @param pairVec are the queue records.

     @param bot indexes the current bottom.

     @return popped value.
  */
  static unsigned int slotPop(BHPair pairVec[],
                              int bot);
};

#endif
