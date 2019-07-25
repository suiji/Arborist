// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontier.h

   @brief Partitions tree frontier, typically breadth-first.

   @author Mark Seligman

 */

#ifndef PARTITION_FRONTIER_H
#define PARTITION_FRONTIER_H

#include "splitcoord.h"
#include "sumcount.h"
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
  IndexType splitIdx; // Unique level identifier.
  IndexType ptId; // Index of associated PTSerial node.
  IndexType lhStart; // Start position of LH in buffer:  Swiss cheese.
  IndexType extent; // # distinct indices in the set.
  IndexType sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  IndexType relBase; // Local copy of frontier's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Set iff argMax nontrivial.)
  bool doesSplit; // iff local conditions satisfied.
  bool unsplitable;  // Candidate found to have single response value.
  IndexType lhExtent; // Total indices over LH.
  IndexType lhSCount; // Total samples over LH.
  double sumExpl; // Sum of explicit index responses.

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  IndexType ptExpl;
  IndexType ptImpl;
  IndexType succExpl; // Fixed:  level index of explicit successor, if any.
  IndexType succImpl; // Fixed:  " " implicit " "
  IndexType offExpl; // Increases:  accumulating explicit offset.
  IndexType offImpl; // Increases:  accumulating implicit offset.
  unsigned char pathExpl;  // Fixed:  path to explicit successor, if any.
  unsigned char pathImpl; // Fixed:  path to implicit successor, if any.
  vector<SumCount> ctgExpl; // Per-category sums.
  bool leftExpl; // Fixed:  whether left split explicit (else right).

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.

  
  /**
     @brief Initializes certain fields to a default terminal state.

     @param inatt is an inattainable value.
   */
  void initInattainable(IndexType inatt) {
    succExpl = succImpl = offExpl = offImpl = inatt;
  }
  
  /**
     @brief Initializes index set as a successor node.
  */
  void succInit(class Frontier *frontier,
                class Bottom *bottom,
                const class PreTree* preTree,
                const IndexSet* par,
                bool isLeft);

  
  void nontermReindex(const class BV *replayExpl,
                      class Frontier *index,
                      IndexType idxLive,
                      vector<IndexType> &succST);
  
 public:
  IndexSet();


  /**
     @brief Initializes root set using sample summary.

     @param sample summarizes the tree's response sampling.
   */
  void initRoot(const class Sample* sample);


  void decr(vector<SumCount> &_ctgTot,
            const vector<SumCount> &_ctgSub);

  
  /**
     @brief Consumes iSet contents into pretree or terminal map.
  */
  void consume(class Frontier *indexlevel,
               const class SplitFrontier* splitFrontier,
               class PreTree *preTree);

  /**
     @brief Caches state necessary for reindexing and useful subsequently.
  */
  void nonterminal(class Frontier *frontier,
                   const class SplitFrontier* splitFrontier,
                   class PreTree *preTree);

  
  inline void nonterminal(double minInfo,
                          IndexType lhSCount,
                          IndexType lhExtent) {
    this->minInfo = minInfo;
    this->lhSCount = lhSCount;
    this->lhExtent = lhExtent;
  }
  
  /**
     @brief Dispatches index set to frontier.

     @param frontier holds the partitioned data.
  */
  void terminal(class Frontier *frontier);


  void blockReplay(class Frontier* frontier,
                   const IndexRange& range);

  
  
  /**
     @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
  */
  void reindex(const class BV *replayExpl,
               class Frontier *index,
               IndexType idxLive,
               vector<IndexType> &succST);


  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares outputs the response sum of squares, over categories.

     @return per-category sums for the node.
  */
  vector<double> sumsAndSquares(double& sumSquares);

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
                Frontier* frontier,
                const class PreTree* preTree,
                bool isLeft) const;

  
  /**
     @param Determines pretree index of specified successor.

     @return pretree index determined.
   */
  IndexType getPTIdSucc(const class PreTree* preTree,
                        bool isLeft) const;

  
  
  /**
     @brief Getter for split index.
   */
  inline auto getSplitIdx() const {
    return splitIdx;
  }


  /**
     @brief Determines whether specified hand of split is explicit.

     @return true iff this is the explicit hand.
   */
  inline bool isExplHand(bool isLeft) const {
    return leftExpl ? isLeft : !isLeft;
  }

  
  inline const vector<SumCount>& getCtgSum() const {
    return ctgSum;
  }


  inline const vector<SumCount>& getCtgExpl() const {
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
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline auto getMinInfo() const {
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
  inline IndexType offspring(bool expl,
                             IndexType& pathSucc,
                             IndexType& ptSucc) {
    IndexType iSetSucc;
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
  inline IndexType offspring(bool expl,
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
class Frontier {
  static unsigned int minNode;
  static unsigned int totLevels;
  unique_ptr<class ObsPart> obsPart;
  vector<IndexSet> indexSet;
  const IndexType bagCount;
  unique_ptr<class SplitFrontier> splitFrontier;
  unique_ptr<class Bottom> bottom;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  bool levelTerminal; // Whether this level must exit.
  unsigned int idxLive; // Total live indices.
  unsigned int liveBase; // Accumulates live index offset.
  unsigned int extinctBase; // Accumulates extinct index offset.
  unsigned int succLive; // Accumulates live indices for upcoming level.
  unsigned int succExtinct; // " " extinct "
  vector<IndexType> relBase; // Node-to-relative index.
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
  vector<IndexSet> splitDispatch(class PreTree* preTree,
                                 unsigned int level);

  /**
     @brief Counts offspring of this node, assumed not to be a leaf.

     @return count of offspring nodes.
  */
  unsigned int splitCensus(const IndexSet& iSet,
                           IndexType& idxMax);

  /**
     @brief Establishes splitting parameters for next frontier level.

     @param[out] idxMax is the index high watermark of the next level.

     @return the number of splitable nodes in the next level.
   */
  IndexType nextLevel(IndexType& idxMax);
  
  /**
     @brief Consumes current level of splits into crescent tree and sets repartitioning bits.

     @param preTree represents the crescent tree.

     @param idxMax is the maximum live index value.

     @param splitNext is the number of splits in the new level.

     Remaining parameters as described above.
  */
  void consume(class PreTree *preTree,
               IndexType splitNext);

  /**
     @brief Reindexes by level modes: node-relative, subtree-relative, mixed.

     Parameters as above.
   */
  void reindex(IndexType idxMax,
               IndexType splitNext);

  
  /**
     @brief Produces new level's index sets and dispatches extinct nodes to pretree frontier.

     Parameters as described above.

     @return next level's splitable index set.
  */
  vector<IndexSet> produce(const class PreTree *preTree,
                           IndexType splitNext);


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
  Frontier(const class SummaryFrame* frame,
             const class Sample* sample);

  ~Frontier();

  /**
    @brief Trains one tree.

    @param summaryFrame contains the predictor type mappings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static unique_ptr<class PreTree> oneTree(const class SummaryFrame* frame,
                                           const class Sample* sample);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
  */
  void levels(class PreTree* preTree);
  

  /**
     @brief Accumulates index parameters of successor level.

     @param succExent is the index of extent of the putative successor set.

     @param[out] idxMax outputs the maximum successor index.

     @return count of splitable sets precipitated in next level:  0 or 1.
  */
  unsigned int splitAccum(IndexType succExtent,
                          IndexType& idxMax);

  /**
     @brief Builds index base offsets to mirror crescent pretree level.

     @param extent is the count of the index range.

     @param ptId is the index of the corresponding pretree node.

     @param offOut outputs the node-relative starting index.  Should not
     exceed 'idxExtent', the live high watermark of the previous level.

     @param terminal is true iff predecessor node is terminal.

     @return successor index count.
  */
  IndexType idxSucc(IndexType extent,
                       IndexType ptId,
                       IndexType &outOff,
                       bool terminal = false);


  /**
     @brief Repartitions sample map for a block of indices.

     @param range is the range of indices defining the block.

     Passes through to ObsPart method.
  */
  double blockReplay(const IndexSet* iSet,
                     const IndexRange& range,
                     vector<SumCount>& ctgExpl) const;


  /**
     @brief Drives node-relative re-indexing.
   */
  void nodeReindex();

  /**
     @brief Subtree-relative reindexing:  indices randomly distributed
     among nodes (i.e., index sets).
  */
  void subtreeReindex(IndexType splitNext);

  /**
     @brief Updates the split/path/pretree state of an extant index based on
     its position in the next level (i.e., left/right/extinct).

     @param stPath is a subtree-relative path.
  */
  void chunkReindex(class IdxPath *stPath,
                    IndexType splitNext,
                    IndexType chunkStart,
                    IndexType chunkNext);

  /**
     @brief As above, but initializes node-relative mappings for subsequent
     levels.  Employs accumulated state and cannot be parallelized.
  */
  void transitionReindex(IndexType splitNext);

  /**
     @brief Updates the mapping from live relative indices to associated
     PreTree indices.

     @return corresponding subtree-relative index.
  */
  IndexType relLive(IndexType relIdx,
                       IndexType targIdx,
                       IndexType path,
                       IndexType base,
                       IndexType ptIdx);
  /**
     @brief Translates node-relative back to subtree-relative indices on 
     terminatinal node.

     @param relIdx is the node-relative index.

     @param ptId is the pre-tree index of the associated node.
  */
  void relExtinct(IndexType relIdx,
                  IndexType ptId);

  
  /**
     @brief Visits all live indices, so likely worth parallelizing.

     @param[out] ctgSum outputs the per-category sum of responses, per node.

     @return category-summed squares of responses, per node.
  */
  vector<double> sumsAndSquares(vector<vector<double> >&ctgSum);


  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool isSplitable(IndexType extent) const {
    return !levelTerminal && extent >= minNode;
  }


  /**
     @brief Getter for # of distinct in-bag samples.

     @return bagCount value.
   */
  inline auto getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for IndexSet at a given coordinate.

     @param splitCoord is the coordinate.

     @return reference to set at coordinate.
   */
  inline const IndexSet& getISet(const SplitCoord& splitCoord) const {
    return indexSet[splitCoord.nodeIdx];
  }
  
  /**
     @brief Accessor for count of splitable sets.
   */
  inline IndexType getNSplit() const {
    return indexSet.size();
  }

  /**
     @brief Accessor for sum of sampled responses over set.

     @param splitIdx is the level-relative index of a set.

     @return index set's sum value.
   */
  inline auto getSum(IndexType splitIdx) const {
    return indexSet[splitIdx].getSum();
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCount(IndexType splitIdx) const {
    return indexSet[splitIdx].getSCount();
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline auto getExtent(IndexType splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  /**
     @brief Accessor for relative base of split.
   */
  inline auto getRelBase(IndexType splitIdx) const {
    return relBase[splitIdx];
  }


  /**
     @brief Indicates whether index set is inherently unsplitable.
   */
  inline bool isUnsplitable(IndexType splitIdx) const {
    return indexSet[splitIdx].isUnsplitable();
  }

  
  /**
     @brief Dispatches consecutive node-relative indices to frontier map for
     final pre-tree node assignment.
  */
  void relExtinct(IndexType relBase,
                  IndexType extent,
                  IndexType ptId) {
    for (IndexType relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      relExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush() {
    if (nodeRel) {
      for (IndexType relIdx = 0; relIdx < idxLive; relIdx++) {
        relExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
