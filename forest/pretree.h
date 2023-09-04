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
#include "decnode.h"
#include "samplemap.h"

#include <vector>


/**
   @brief Serialized representation of the pre-tree, suitable for tranfer between devices such as coprocessors, disks and compute nodes.
*/
class PreTree {
  static IndexT leafMax; // User option:  maximum # leaves, if > 0.
  IndexT leafCount; // Running count of leaves.
  vector<DecNode> nodeVec; // Vector of tree nodes.
  vector<double> scores;
  vector<double> infoLocal; //< Per-predictor split information.
  vector<double> infoNode; ///< Per-node " ".  Leaf merging onlye.
  BV splitBits; // Bit encoding of factor splits.
  BV observedBits; // Bit encoding of factor values.
  size_t bitEnd; // Next free slot in either bit vector.
  SampleMap terminalMap;


  /**
     @brief Consumes each criterion in a collection.

     @param critVec collects splits defining criteria.
   */
  void consumeCriteria(const class SplitFrontier* sf,
		       const vector<class SplitNux>& critVec);


  /**
     @brief Enumerates leaves.

     Leaf ordering is currently irrelevant, from the perspective of
     prediction, as support for premature exit is not required.  Post-
     training adjustments to the tree, however, require the ability to
     reconstruct sample maps at arbitrary locations.  For this reason, a
     depth-first ordering is applied.
   */
  void setLeafIndices();


  
 public:
  /**
   */
  PreTree(const class PredictorFrame* frame,
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

     @param predInfo accumulates the local information contribution.

     @return leaf map from consumed frontier.
  */
  void consume(class Grove* train,
	       struct Leaf* leaf) const;


  /**
     @brief Sets score at index to value passed.
   */
  void setScore(const class IndexSet& iSet,
		double score);

  
  double getScore(IndexT idx) const {
    return scores[idx];
  }


  /**
     @brief Assigns scores to all nodes in the map.
   */
  void scoreNodes(const class Sampler* sampler,
		  const struct SampleMap& map);


  /**
     @brief Caches terminal map, merges, numbers leaves.

     @param smTerminal is the terminal map produce by Frontier.
   */
  void setTerminals(const class SampledObs* sampledObs,
		    SampleMap smTerminal);
  

  /**
     @brief Combines leaves exceeding a specified maximum count.
   */
  void leafMerge();
  

  inline IndexT getHeight() const {
    return nodeVec.size();
  }
  

  inline void resetTerminal(IndexT ptId) {
    nodeVec[ptId].resetTerminal();
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


  inline IndexT getDelIdx(IndexT ptId) const {
    return nodeVec[ptId].getDelIdx();
  }


  /**
     @brief Obtains leaf index of node assumed to be nonterminal.
   */
  inline IndexT getLeafIdx(IndexT ptIdx) const {
    return nodeVec[ptIdx].getLeafIdx();
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


  DecNode& getNode(IndexT ptId) {
    return nodeVec[ptId];
  }
  

  /**
     @brief Accounts for a block of new criteria or singleton root node.

     Pre-existing terminal node converted to nonterminal for leading criterion.

     @param nCrit is the number of criteria in the block; zero iff block preallocated.
  */
  inline void offspring(IndexT nCrit, bool root = false) {
    if (nCrit > 0 || root) {
      DecNode node;
      nodeVec.insert(nodeVec.end(), nCrit + 1, node);
      scores.insert(scores.end(), nCrit + 1, 0.0);
      infoNode.insert(infoNode.end(), nCrit + 1, 0.0);
      leafCount++; // Two new terminals, minus one for conversion of lead criterion.
    }
  }
};


/**
   @brief Augments a decision node with values to facilitate merging.
 */
struct PTMerge {
  FltVal infoDom; ///< sum of dominated info values.
  IndexT ptId; ///< node id.
};


/**
   @brief Information-base comparator for queue ordering.

   Nodes order is increasing with sum, guaranteeing that offspring are chosen
   before parents.
*/
class InfoCompare {
public:
  bool operator() (const PTMerge& a, const PTMerge& b) {
    return a.infoDom > b.infoDom;
  }
};


#endif
