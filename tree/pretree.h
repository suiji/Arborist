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

#include "bv.h"
#include "typeparam.h"
#include "forestcresc.h"
#include "decnode.h"

#include <vector>

/**
  @brief Decision node specialized for training.

  Information member consumed during forest production.
 */
struct PTNode {
private:
  FltVal info;  // Zero iff terminal.
  DecNode decNode;

public:
  /**
     @brief Constructor.  Decision node set to terminal by its constructor.
   */
  PTNode() : info(0.0) {
  }


  static void setNonterminal(vector<PTNode>& nodeVec,
			     const class SplitNux* nux,
			     IndexT targ);

  
  auto getPredIdx() const {
    return decNode.getPredIdx();
  }
  

  auto getIdFalse(IndexT ptIdx) const {
    return decNode.getIdFalse(ptIdx);
  }


  auto getIdTrue(IndexT ptIdx) const {
    return decNode.getIdTrue(ptIdx);
  }


  bool isNonterminal() const {
    return decNode.isNonterminal();
  }


  void setDelIdx(IndexT ptIdx) {
    decNode.setDelIdx(ptIdx);
  }


  void critBits(const class SplitNux* nux,
		size_t bitEnd) {
    decNode.critBits(nux, bitEnd);
  }


  auto getBitOffset() const {
    return decNode.getBitOffset();
  }

  
  void critCut(const class SplitNux* nux,
	       const class SplitFrontier* splitFrontier) {
    decNode.critCut(nux, splitFrontier);
  }


  void setTerminal() {
    decNode.setTerminal();
  }
  
  
  /**
     @brief Consumes the node fields of nonterminals (splits).

     @param forest[in, out] accumulates the growing forest node vector.
  */
  void consumeNonterminal(ForestCresc<DecNode>* forest,
                          vector<double>& predInfo,
                          IndexT idx) const {
    if (isNonterminal()) {
      forest->nonterminal(idx, decNode);
      predInfo[getPredIdx()] += info;
    }
  }


  /**
     @brief Sets node to nonterminal.

     @param nux contains the splitting information.

     @param height is the current tree height.
   */
  inline void setNonterminal(const SplitNux* nux,
                             IndexT height);
};


/**
   @brief Serialized representation of the pre-tree, suitable for tranfer between devices such as coprocessors, disks and nodes.
*/
class PreTree {
  static IndexT heightEst;
  static IndexT leafMax; // User option:  maximum # leaves, if > 0.
  const IndexT bagCount;
  IndexT height;
  IndexT leafCount;
  size_t bitEnd; // Next free slot in factor bit vector.
  vector<PTNode> nodeVec; // Vector of tree nodes.
  class BV* splitBits;
  vector<IndexT> termST;

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
     @brief Dispatches nonterminal and offspring.
   */
  void nonterminal(const class SplitNux* nux);


  /**
     @brief As above, but invoked incrementally.

     Assumes offspring already dispatched.
   */
  void nonterminalInc(const class SplitNux* nux);
  
  
  /**
     @brief Appends criterion for bit-based branch.

     @param nux summarizes the criterion bits.

     @param cardinality is the predictor's cardinality.

     @param bitsTrue are the bit positions taking the true branch.
  */
  void critBits(const class SplitNux* nux,
                PredictorT cardinality,
		const vector<PredictorT> bitsTrue);

  
  /**
     @brief Appends criterion for cut-based branch.
     
     @param nux summarizes the the cut.
  */
  void critCut(const class SplitNux* nux,
	       const class SplitFrontier* splitFrontier);

  
  /**
     @brief Consumes all pretree nonterminal information into crescent forest.

     @param forest grows by producing nodes and splits consumed from pre-tree.

     @param tIdx is the index of the tree being consumed/produced.

     @param predInfo accumulates the information contribution of each predictor.

     @return leaf map from consumed frontier.
  */
  const vector<IndexT> consume(ForestCresc<DecNode> *forest,
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


  IndexT leafMerge();

  
  /**
     @brief Absorbs the terminal list and merges, if requested.

     Side-effects the frontier map.

     @param stTerm are subtree-relative indices.  These must be mapped to
     sample indices if the subtree is proper.
  */
  void finish(const vector<IndexT>& stTerm);


  inline IndexT getHeight() const {
    return height;
  }
  
  
  inline IndexT getIdTrue(IndexT ptId) const {
    return nodeVec[ptId].getIdTrue(ptId);
  }

  
  inline IndexT getIdFalse(IndexT ptId) const {
    return nodeVec[ptId].getIdFalse(ptId);
  }


  inline IndexT getSuccId(IndexT ptId, bool isLeft) const {
    return isLeft ? nodeVec[ptId].getIdTrue(ptId) : nodeVec[ptId].getIdFalse(ptId);
  }


  /**
     @brief Obtains true and false branch target indices.
   */
  inline void getSuccTF(IndexT ptId,
                        IndexT& ptLeft,
                        IndexT& ptRight) const {
    ptLeft = nodeVec[ptId].getIdTrue(ptId);
    ptRight = nodeVec[ptId].getIdFalse(ptId);
  }
  
  
  /**
     @return true iff node is nonterminal.
   */
  inline bool isNonterminal(IndexT ptId) const {
    return nodeVec[ptId].isNonterminal();
  }


    /**
       @brief Determines whether a nonterminal can be merged with its
       children.

       @param ptId is the index of a nonterminal.

       @return true iff node has two leaf children.
    */
  inline bool isMergeable(IndexT ptId) const {
    return !isNonterminal(getIdTrue(ptId)) && !isNonterminal(getIdFalse(ptId));
  }  

  

  /**
     @brief Accounts for a block of new criteria.

     Pre-existing placeholder node for leading criterion converted to nonterminal.

     @param nCrit is the number of criteria in the block.
  */
  inline void offspring(IndexT nCrit) {
    height += nCrit + 1; // Two new terminals plus nCrit - 1 new nonterminals.
    leafCount++; // Two new terminals, minus one for conversion of lead criterion.
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
  bool descTrue; // Whether this is true-branch descendant of some node.

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
