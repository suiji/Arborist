// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file pretree.h

   @brief Builds a single decision tree and dispatches to crescent forest.

   @author Mark Seligman

 */

#ifndef FOREST_PRETREE_H
#define FOREST_PRETREE_H

#include "bv.h"
#include "typeparam.h"
#include "forest.h"
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

  
  auto getDelIdx() const {
    return decNode.getDelIdx();
  }
  
  
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

  
  void critBits(const class SplitNux& nux,
		size_t bitEnd) {
    decNode.critBits(&nux, bitEnd);
  }


  auto getBitOffset() const {
    return decNode.getBitOffset();
  }

  
  void critCut(const class SplitNux& nux,
	       const class SplitFrontier* splitFrontier) {
    decNode.critCut(&nux, splitFrontier);
  }

  
  /**
     @brief Resetter for merged nonterminals.
   */
  void setTerminal(IndexT leafIdx = 0) {
    decNode.setLeaf(leafIdx);
  }
  
  
  /**
     @brief Consumes the node fields of nonterminals (splits).

     @param forest[in, out] accumulates the growing forest node vector.
  */
  void consume(Forest* forest,
	       vector<double>& predInfo,
	       IndexT idx,
	       IndexT& leafIdx) {
    if (isNonterminal()) {
      predInfo[getPredIdx()] += info;
    }
    else {
      setTerminal(leafIdx++);
    }
    forest->nodeProduce(idx, decNode);
  }

  /**
     @brief Sets node to nonterminal.

     @param nux contains the splitting information.

     @param height is the current tree height.
   */
  inline void setNonterminal(const SplitNux& nux,
                             IndexT height);
};


/**
   @brief Serialized representation of the pre-tree, suitable for tranfer between devices such as coprocessors, disks and compute nodes.
*/
class PreTree {
  static IndexT leafMax; // User option:  maximum # leaves, if > 0.
  IndexT height; // Running count of nodes.
  IndexT leafCount; // Running count of leaves.
  vector<PTNode> nodeVec; // Vector of tree nodes.
  vector<double> scores;
  class BV splitBits; // Bit encoding of factor splits.
  size_t bitEnd; // Next free slot in factor bit vector.
  vector<IndexT> sampleMap; // Frontier mapping of sIdx to ptIdx. EXIT

  
  /**
     @brief Assigns frontier node samples to leaf indices.

     @return map from sample indices to leaf indices.
  */
  const vector<IndexT> sample2Leaf() const;

 public:
  /**
   */
  PreTree(PredictorT cardExtent,
	  IndexT bagCount_);


  /**
   @brief Caches the row count and computes an initial estimate of node count.

   @param leafMax is a user-specified limit on the number of leaves.
 */
  static void init(IndexT leafMax_);


  static void deInit();

  
  /**
     @brief Verifies that frontier samples all map to leaf nodes.

     @return count of non-leaf nodes encountered.
   */
  IndexT checkFrontier(const vector<IndexT>& stMap) const;


  /**
     @brief Consumes a collection of compound criteria.
   */
  void consumeCompound(const class SplitFrontier* sf,
		       const vector<vector<SplitNux>>& nuxMax);

  
  /**
     @brief Consumes each criterion in a collection.

     @param critVec collects splits defining criteria.

     @param compound is true iff collection is compound.
   */
  void consumeCriteria(const class SplitFrontier* sf,
		       const vector<class SplitNux>& critVec);


  /**
     @brief Dispatches nonterminal and offspring.

     @param preallocate indicates whether criteria block has been preallocated.
   */
  void addCriterion(const class SplitFrontier* sf,
		    const class SplitNux& nux,
		    bool preallocated = false);


  /**
     @brief Appends criterion for bit-based branch.

     @param nux summarizes the criterion bits.

     @param cardinality is the predictor's cardinality.

     @param bitsTrue are the bit positions taking the true branch.
  */
  void critBits(const class SplitFrontier* sf,
		const class SplitNux& nux);

  
  /**
     @brief Appends criterion for cut-based branch.
     
     @param nux summarizes the the cut.
  */
  void critCut(const class SplitFrontier* sf,
	       const class SplitNux& nux);

  
  /**
     @brief Consumes all pretree nonterminal information into crescent forest.

     @param forest grows by producing nodes and splits consumed from pre-tree.

     @param predInfo accumulates the information contribution of each predictor.

     @return leaf map from consumed frontier.
  */
  const vector<IndexT> consume(Forest *forest,
			       vector<double> &predInfo);

  
  /**
     @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.

     Leaf information is post-assigned by the response-dependent Sample methods.

     @param[in, out]  forest inputs/outputs the updated forest.

     @param[out] predInfo outputs the predictor-specific information values.
  */
  void consumeNodes(Forest *forest,
		    vector<double> &predInfo);


  void setScore(const class SplitFrontier* sf,
		const class IndexSet& iSet);


  /**
     @brief Assigns scores to all nodes in the map.
   */
  void scoreNodes(const class Sampler* sampler,
		  const class SampleMap& map);

  /**
     @brief Obtains scores assigned to indices in the map.
   */
  void setTerminals(const SampleMap& terminalMap);
  

  IndexT leafMerge();
  

  inline IndexT getHeight() const {
    return height;
  }
  

  inline void setTerminal(IndexT ptId) {
    nodeVec[ptId].setTerminal();
  }

  
  inline IndexT getIdTrue(IndexT ptId) const {
    return nodeVec[ptId].getIdTrue(ptId);
  }

  
  inline IndexT getIdFalse(IndexT ptId) const {
    return nodeVec[ptId].getIdFalse(ptId);
  }


  inline IndexT getSuccId(IndexT ptId, bool senseTrue) const {
    return senseTrue ? nodeVec[ptId].getIdTrue(ptId) : nodeVec[ptId].getIdFalse(ptId);
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

     Pre-existing terminal node converted to nonterminal for leading criterion.

     @param nCrit is the number of criteria in the block; zero iff block preallocated.
  */
  inline void offspring(IndexT nCrit) {
    if (nCrit > 0) {
      height += nCrit + 1; // Two new terminals plus nCrit - 1 new nonterminals.
      leafCount++; // Two new terminals, minus one for conversion of lead criterion.
    }
  }
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
