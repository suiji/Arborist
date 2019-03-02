// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file index.h

   @brief Definitions for classes maintaining the index-tree representation.

   @author Mark Seligman

 */

#ifndef ARBORIST_INDEX_H
#define ARBORIST_INDEX_H

#include "typeparam.h"
#include <vector>


/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexSets of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexSets only live within a single level.
*/
class IndexSet {
  unsigned int splitIdx; // Unique level identifier.
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned int lhStart; // Start position of LH in buffer:  Swiss cheese.
  unsigned int extent; // # distinct indices in the set.
  unsigned int sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  unsigned int relBase; // Local copy of indexLevel's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<class SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Set iff argMax nontrivial.)
  bool doesSplit; // iff argMax nontrivial.
  bool unsplitable;  // Candidate found to have single response value.
  unsigned int lhExtent; // Total indices over LH.
  unsigned int lhSCount; // Total samples over LH.
  double sumExpl; // Sum of explicit index responses.

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  unsigned int ptExpl;
  unsigned int ptImpl;
  unsigned int succExpl; // Fixed:  level index of explicit successor, if any.
  unsigned int succImpl; // Fixed:  " " implicit " "
  unsigned int offExpl; // Increases:  accumulating explicit offset.
  unsigned int offImpl; // Increases:  accumulating implicit offset.
  unsigned char pathExpl;  // Fixed:  path to explicit successor, if any.
  unsigned char pathImpl; // Fixed:  path to implicit successor, if any.
  vector<class SumCount> ctgExpl; // Per-category sums.
  bool leftExpl; // Fixed:  whether left split explicit (else right).

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.

  
  /**
     @brief Initializes certain fields to a default terminal state.

     @param inatt is an inattainable value.
   */
  void initInattainable(unsigned int inatt) {
    succExpl = succImpl = offExpl = offImpl = inatt;
  }
  
  /**
     @brief Initializes index set as a successor node.
  */
  void succInit(class IndexLevel *indexLevel,
                class Bottom *bottom,
                const class PreTree* preTree,
                const IndexSet* par,
                bool isLeft);

  
  void nontermReindex(const class BV *replayExpl,
                      class IndexLevel *index,
                      unsigned int idxLive,
                      vector<unsigned int> &succST);
  
 public:
  IndexSet();


  /**
     @brief Initializes root set using sample summary.

     @param sample summarizes the tree's response sampling.
   */
  void initRoot(const class Sample* sample);


  void decr(vector<class SumCount> &_ctgTot, const vector<class SumCount> &_ctgSub);

  /**
     @brief Absorbs parameters of informative splits.

     @param argMax contains the successful splitting candidates.
  */
  void applySplit(const vector<class SplitCand> &argMax);
  
  /**
     @brief Consumes relevant contents of split signature, if any, and accumulates leaf and splitting census.

     @param splitNext counts splitable nodes precipitated in the next level.
  */
  void splitCensus(class IndexLevel *indexLevel,
                   unsigned int &leafThis,
                   unsigned int &splitNext,
                   unsigned int &idxLive,
                   unsigned int &idxMax);

  
  /**
     @brief Consumes iSet contents into pretree or terminal map.
  */
  void consume(class IndexLevel *indexlevel,
               const class Run* run,
               class PreTree *preTree,
               const vector<class SplitCand> &argMax);

  /**
     @brief Caches state necessary for reindexing and useful subsequently.
  */
  void nonTerminal(class IndexLevel *indexLevel,
                   const class Run* run,
                   class PreTree *preTree,
                   const class SplitCand &argMax);

  /**
     @brief Dispatches index set to frontier.
  */
  void terminal(class IndexLevel *indexLevel);

  /**
     @brief Directs split-based repartitioning and precipitates creation of a branch node.

     Remaining parameters as described above.

     @return true iff left hand of the split is explicit.
  */
  bool branchNum(const class SplitCand& argMax,
                 class PreTree* preTree,
                 class IndexLevel* indexLevel);

  
  void blockReplay(const class SplitCand& argMax,
                   unsigned int blockStart,
                   unsigned int blockExtent,
                   class IndexLevel* indexLevel);

  /**
     @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
  */
  void reindex(const class BV *replayExpl,
               class IndexLevel *index,
               unsigned int idxLive,
               vector<unsigned int> &succST);


  /**
     @brief Accumulates index parameters of successor level.

     @param succExent is the index of extent of the putative successor set.

     @param[out] idxLive outputs the number of live successor indices.

     @param[out] idxMax outputs the maximum successor index.

     @return count of splitable sets precipitated in next level:  0/1.
  */
  static unsigned splitAccum(class IndexLevel *indexLevel,
                             unsigned int succExtent,
                             unsigned int &idxLive,
                             unsigned int &idxMax);

  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares accumulates the sum of squares over each category.
     Assumed intialized to zero.

     @param[in, out] sumOut records the response sums, by category.  Assumed initialized to zero.
  */
  void sumsAndSquares(double &sumSquares, double *sumOut);

  bool isUnsplitable() const {
    return unsplitable;
  }


  /**
     @brief Produces next level's iSets for given hand (LH or RH) of a split.

     @param indexNext is the crescent successor level of index sets.

     @param isLeft is true iff this is the LH successor.
  */
  void succHand(vector<IndexSet>& indexNext,
                class Bottom* bottom,
                IndexLevel* indexLevel,
                const class PreTree* preTree,
                bool isLeft) const;

  
  /**
     @param Determines pretree index of specified successor.

     @return pretree index determined.
   */
  unsigned int getPTIdSucc(const class PreTree* preTree,
                           bool isLeft) const;

  
  
  /**
     @brief Getter for split index.
   */
  inline unsigned int getSplitIdx() const {
    return splitIdx;
  }


  /**
     @brief Determines whether specified hand of split is explicit.

     @return true iff this is the explicit hand.
   */
  inline bool isExplHand(bool isLeft) const {
    return leftExpl ? isLeft : !isLeft;
  }

  
  inline const vector<class SumCount>& getCtgSum() const {
    return ctgSum;
  }


  inline const vector<class SumCount>& getCtgExpl() const {
    return ctgExpl;
  }


  inline auto getIdxSucc(bool isLeft) const {
    return isExplHand(isLeft) ? succExpl : succImpl;
  }


  inline auto getSumSucc(bool isLeft) const {
    return isExplHand(isLeft) ? sumExpl : sum - sumExpl;
  }


  inline auto getPathSucc(bool isLeft) const {
    return isExplHand(isLeft) ? pathExpl : pathImpl;
  }

  
  inline auto getSCountSucc(bool isLeft) const {
    return isLeft ? lhSCount : sCount - lhSCount;
  }

  inline auto getLHStartSucc(bool isLeft) const {
    return isLeft ?  lhStart : lhStart + lhExtent;
  }


  inline auto getExtentSucc(bool isLeft) const {
    return isLeft ? lhExtent : extent - lhExtent;
  }
  
  /**
     @brief Getters returning like-named member value.
   */

  inline auto getStart() const {
    return lhStart;
  }

  
  inline auto getExtent() const {
    return extent;
  }


  inline auto getSum() const {
    return sum;
  }
  

  inline auto getSCount() const {
    return sCount;
  }


  inline auto getPTId() const {
    return ptId;
  }

  
  /**
     @brief Copies certain fields of an index set to a splitting candidate.

     @param[in, out] cand is the splitting candidate.

     @return extent of index set specified by candidate.
   */
  unsigned int setCand(class SplitCand* cand) const;


  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double getMinInfo() const {
    return minInfo;
  }


  /**
     @brief L/R accessor for subtree-relative reindexing.

     @param expl is true iff the successor lies in the explicit side of
     the split.

     @param pathSucc outputs the (possibly pseudo) successor path.

     @param idxSucc outputs the (possibly pseudo) successor index.

     @return index (possibly pseudo) of successor index set.
   */
  inline unsigned int offspring(bool expl,
                                unsigned int &pathSucc,
                                unsigned int &ptSucc) {
    unsigned int iSetSucc;
    if (!doesSplit) {  // Terminal from previous level.
      iSetSucc = succOnly;
      ptSucc = ptId;
      pathSucc = 0; // Dummy:  overwritten by caller.
    }
    else {
      iSetSucc = expl ? succExpl : succImpl;
      pathSucc = expl ? pathExpl : pathImpl;
      ptSucc = expl ? ptExpl : ptImpl;
    }
    return iSetSucc;
  }

  
  /**
     @brief As above, but also tracks (pseudo) successor indices.  State
     is side-effected, moreover, so must be invoked sequentially.
   */
  inline unsigned int offspring(bool expl,
                                unsigned int &pathSucc,
                                unsigned int &idxSucc,
                                unsigned int &ptSucc) {
    idxSucc = !doesSplit ? offOnly++ : (expl ? offExpl++ : offImpl++);
    return offspring(expl, pathSucc, ptSucc);
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class IndexLevel {
  static unsigned int minNode;
  static unsigned int totLevels;
  unique_ptr<class SamplePred> samplePred;
  vector<IndexSet> indexSet;
  const unsigned int bagCount;
  unique_ptr<class Bottom> bottom;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  bool levelTerminal; // Whether this level must exit.
  unsigned int idxLive; // Total live indices.
  unsigned int liveBase; // Accumulates live index offset.
  unsigned int extinctBase; // Accumulates extinct index offset.
  unsigned int succLive; // Accumulates live indices for upcoming level.
  unsigned int succExtinct; // " " extinct "
  vector<unsigned int> relBase; // Node-to-relative index.
  vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  vector<unsigned int> rel2ST; // Maps to subtree index.
  vector<unsigned int> rel2PT; // Maps to pretree index.
  vector<unsigned int> st2Split; // Useful for subtree-relative indexing.
  vector<unsigned int> st2PT; // Frontier map.
  unique_ptr<class BV> replayExpl; // Per-sample partition direction:  L/R.

  /**
     @brief Applies splitting results to new level.

     @param argMax are the per-node splitting candidates.

     @param levelTerminal_ indicates whether new level marked as final.
  */
  void splitDispatch(const class SplitNode* splitNode,
                     const vector<class SplitCand> &argMax,
                     class PreTree* preTree,
                     bool levelTerminal_);

  /**
     @brief Consumes current level of splits into crescent tree and sets repartitioning bits.

     @param preTree represents the crescent tree.

     @param splitNext is the number of splits in the new level.

     Remaining parameters as described above.
  */
  void consume(const class SplitNode* splitNode,
               class PreTree *preTree,
               const vector<class SplitCand> &argMax,
               unsigned int splitNext,
               unsigned int leafNext,
               unsigned int idxMax);

  /**
     @brief Produces new level's index sets and dispatches extinct nodes to pretree frontier.

     Parameters as described above.
  */
  void produce(const class PreTree *preTree,
               unsigned int splitNext);


 public:

  /**
     @brief Initializes static invariants.

     @param minNode_ is the minimum node size for splitting.

     @param totLevels_ is the maximum number of levels to evaluate.
  */
  static void immutables(unsigned int minNode_,
                         unsigned int totLevels_);


  /**
     @brief Resets statics to default values.
  */
  static void deImmutables();


  /**
     @brief Per-tree constructor.  Sets up root node for level zero.
  */
  IndexLevel(const class FrameTrain* frameTrain,
             const class RowRank* rowRank,
             const class Sample* sample);

  ~IndexLevel();

  /**
    @brief Trains one tree.

    @param frameTrain contains the predictor type mappings.

    @param rowRank contains the per-predictor observation rankings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static shared_ptr<class PreTree> oneTree(const class FrameTrain* frameTrain,
                                           const class RowRank* rowRank,
                                           const class Sample* sample);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
     
     @return trained pretree object.
  */
  shared_ptr<class PreTree> levels(const class FrameTrain *frameTrain,
                                   const class Sample* sample);
  

  /**
     @brief Builds index base offsets to mirror crescent pretree level.

     @param extent is the count of the index range.

     @param ptId is the index of the corresponding pretree node.

     @param offOut outputs the node-relative starting index.  Should not
     exceed 'idxExtent', the live high watermark of the previous level.

     @param terminal is true iff predecessor node is terminal.

     @return successor index count.
  */
  unsigned int idxSucc(unsigned int extent,
                       unsigned int ptId,
                       unsigned int &outOff,
                       bool terminal = false);

  double blockReplay(const class SplitCand& argMax,
                     vector<SumCount>& ctgExpl);
  
  /**
     @brief Repartitions sample map for a block of indices.
  */
  double blockReplay(const class SplitCand& argMax,
                     unsigned int blockStart,
                     unsigned int blockExtent,
                     vector<SumCount>& ctgExpl) const;

  /**
     @brief Drives node-relative re-indexing.
   */
  void nodeReindex();

  /**
     @brief Subtree-relative reindexing:  indices randomly distributed
     among nodes (i.e., index sets).
  */
  void subtreeReindex(unsigned int splitNext);

  /**
     @brief Updates the split/path/pretree state of an extant index based on
     its position in the next level (i.e., left/right/extinct).

     @param stPath is a subtree-relative path.
  */
  void chunkReindex(class IdxPath *stPath,
                    unsigned int splitNext,
                    unsigned int chunkStart,
                    unsigned int chunkNext);

  /**
     @brief As above, but initializes node-relative mappings for subsequent
     levels.  Employs accumulated state and cannot be parallelized.
  */
  void transitionReindex(unsigned int splitNext);

  /**
     @brief Updates the mapping from live relative indices to associated
     PreTree indices.

     @return corresponding subtree-relative index.
  */
  unsigned int relLive(unsigned int relIdx,
                       unsigned int targIdx,
                       unsigned int path,
                       unsigned int base,
                       unsigned int ptIdx);
  /**
     @brief Translates node-relative back to subtree-relative indices on 
     terminatinal node.

     @param relIdx is the node-relative index.

     @param ptId is the pre-tree index of the associated node.
  */
  void relExtinct(unsigned int relIdx, unsigned int ptId);

  
  /**
     @brief Visits all live indices, so likely worth parallelizing.
     TODO:  Build categorical sums within Replay().
  */
  void sumsAndSquares(unsigned int ctgWidth,
                      vector<double> &sumSquares,
                      vector<double> &ctgSum);


  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool isSplitable(unsigned int extent) {
    return !levelTerminal && extent >= minNode;
  }


  /**
     @brief Getter for # of distinct in-bag samples.

     @return bagCount value.
   */
  inline unsigned int getBagCount() const {
    return bagCount;
  }


  /**
     @brief Accessor for count of splitable sets.
   */
  inline unsigned int getNSplit() const {
    return indexSet.size();
  }

  /**
     @brief Accessor for sum of sampled responses over set.

     @param splitIdx is the level-relative index of a set.

     @return index set's sum value.
   */
  inline double getSum(unsigned int splitIdx) const {
    return indexSet[splitIdx].getSum();
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline unsigned int getSCount(unsigned int splitIdx) const {
    return indexSet[splitIdx].getSCount();
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline unsigned int getExtent(unsigned int splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  /**
     @brief Copies certain fields of this set to a splitting candidate.

     @param[in, out] is the splitting candidate.

     @return index extent of this set.
   */
  unsigned int setCand(class SplitCand* cand) const;


  /**
     @brief Accessor for relative base of split.
   */
  inline unsigned int getRelBase(unsigned int splitIdx) const {
    return relBase[splitIdx];
  }


  /**
     @brief Indicates whether index set is inherently unsplitable.
   */
  inline bool isUnsplitable(unsigned int splitIdx) const {
    return indexSet[splitIdx].isUnsplitable();
  }

  
  /**
     @brief Dispatches consecutive node-relative indices to frontier map for
     final pre-tree node assignment.
  */
  void relExtinct(unsigned int relBase,
                  unsigned int extent,
                  unsigned int ptId) {
    for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      relExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush() {
    if (nodeRel) {
      for (unsigned int relIdx = 0; relIdx < idxLive; relIdx++) {
        relExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
