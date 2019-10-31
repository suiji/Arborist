// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file pretree.h

   @brief Class defintions for the pre-tree, a serial and minimal representation from which the decision tree is built.

   @author Mark Seligman

 */

#ifndef TREE_PRETREE_H
#define TREE_PRETREE_H

#include "decnode.h"
#include "bv.h"
#include "ptnode.h"
#include "crit.h"
#include "indexset.h"

#include <vector>


/**
   @brief Workspace for merging PTNodes:  copies 'info' and records
   offsets and merge state.
 */
/**
   @brief Serialized representation of the pre-tree, suitable for tranfer between devices such as coprocessors, disks and nodes.
*/

class PreTree {
  static IndexT heightEst;
  static IndexT leafMax; // User option:  maximum # leaves, if > 0.
  const unsigned int bagCount;
  IndexT height;
  IndexT leafCount;
  size_t bitEnd; // Next free slot in factor bit vector.
  vector<PTNode<DecNode>> nodeVec; // Vector of tree nodes.
  vector<struct Crit> crit;
  class BV *splitBits;
  vector<unsigned int> termST;

  /**
     @brief Constructs mapping from sample indices to leaf indices.

     @param[in, out] forest accumulates the growing forest.

     @return rewritten map.
  */
  const vector<IndexT> frontierConsume(ForestCresc<DecNode> *forest) const;

  /**
     @return BV-aligned length of used portion of split vector.
  */
  size_t getBitWidth();


  /**
     @brief Accounts for the addition of two terminals to the tree.

     @return void, with incremented height and leaf count.
  */
  inline void terminalOffspring() {
  // Two more leaves for offspring, one fewer for this.
    height += 2;
    leafCount++;
  }


 public:
  /**
   */
  PreTree(PredictorT cardExtent,
	  IndexT bagCount_);


  /**
  */
  ~PreTree();


  /**
   @brief Caches the row count and computes an initial estimate of node count.

   @param _nSamp is the number of samples.

   @param _minH is the minimal splitable index node size.

   @param leafMax is a user-specified limit on the number of leaves.
 */
  static void immutables(IndexT nSamp, IndexT minH, IndexT leafMax_);


  static void deImmutables();


  /**
     @brief Refines the height estimate using the actual height of a
     constructed PreTree.

     @param height is an actual height value.
  */
  static void reserve(IndexT height);


  /**
     @brief Dispatches nonterminal method according to split type.
   */
  void nonterminal(double info,
                   class IndexSet* iSet);

  
  /**
     @brief Appends criterion for bit-based branch.

     @param predIdx is the criterion predictor.

     @param cardinality is the predictor's cardinality.
  */
  void critBits(const class IndexSet* iSet,
                PredictorT predIdx,
                PredictorT cardinality);

  
  /**
     @brief Appends criterion for cut-based branch.
     
     @param rankRange bounds the cut-defining ranks.
  */
  void critCut(const class IndexSet* iSet,
               PredictorT predIdx,
	       double quantRank);

  
  /**
     @brief Consumes all pretree nonterminal information into crescent forest.

     @param forest grows by producing nodes and splits consumed from pre-tree.

     @param tIdx is the index of the tree being consumed/produced.

     @param predInfo accumulates the information contribution of each predictor.

     @return leaf map from consumed frontier.
  */
  const vector<unsigned int> consume(ForestCresc<DecNode> *forest,
                                     unsigned int tIdx,
                                     vector<double> &predInfo);

  
  /**
     @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.

     Leaf information is post-assigned by the response-dependent Sample methods.

     @param[in, out]  forest inputs/outputs the updated forest.

     @param[out] predInfo outputs the predictor-specific information values.
  */
  void consumeNonterminal(ForestCresc<DecNode> *forest,
                          vector<double> &predInfo);

  
  /**
     @brief Sets specified bit in (left) splitting bit vector.

     @param iSet is the index node for which the LH bit is set.

     @param pos is the bit position beyond to set.
  */
  void setLeft(const class IndexSet* iSet,
               IndexT pos);


  IndexT leafMerge();

  
  /**
     @brief Absorbs the terminal list and merges, if requested.

     Side-effects the frontier map.

     @param stTerm are subtree-relative indices.  These must be mapped to
     sample indices if the subtree is proper.
  */
  void finish(const vector<IndexT>& stTerm);

  
  inline IndexT getLHId(IndexT ptId) const {
    return nodeVec[ptId].getLHId(ptId);
  }

  
  inline IndexT getRHId(IndexT ptId) const {
    return nodeVec[ptId].getRHId(ptId);
  }


  inline IndexT getSuccId(IndexT ptId, bool isLeft) const {
    return isLeft ? nodeVec[ptId].getLHId(ptId) : nodeVec[ptId].getRHId(ptId);
  }
  
  /**
     @return true iff node is nonterminal.
   */
  inline bool isNonTerminal(IndexT ptId) const {
    return nodeVec[ptId].isNonTerminal();
  }


    /**
       @brief Determines whether a nonterminal can be merged with its
       children.

       @param ptId is the index of a nonterminal.

       @return true iff node has two leaf children.
    */
  inline bool isMergeable(IndexT ptId) const {
    return !isNonTerminal(getLHId(ptId)) && !isNonTerminal(getRHId(ptId));
  }  

  
  /**
     @brief Fills in references to values known to be useful for building
     a block of PreTree objects.
   */
  void blockBump(IndexT& _height,
		 IndexT& _maxHeight,
		 size_t& _bitWidth,
		 IndexT& _leafCount,
		 IndexT& _bagCount);
};


template<typename nodeType>
struct PTMerge {
  FltVal info;
  IndexT ptId;
  IndexT idMerged;
  IndexT root;
  IndexT parId;
  IndexT idSib; // Sibling id, if not root else zero.
  bool descLH; // Whether this is left descendant of some node.

  static vector<PTMerge<nodeType>> merge(const PreTree* preTree,
				  IndexT height,
				  IndexT leafDiff);

};


/**
   @brief Information-base comparator for queue ordering.
*/
template<typename nodeType>
class InfoCompare {
public:
  bool operator() (const PTMerge<nodeType>& a, const PTMerge<nodeType>& b) {
    return a.info > b.info;
  }
};


#endif
