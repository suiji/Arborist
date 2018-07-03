// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.h

   @brief Data structures and methods for constructing and walking
       the decision trees.

   @author Mark Seligman
 */

#ifndef ARBORIST_FOREST_H
#define ARBORIST_FOREST_H

#include <vector>
#include <algorithm>

#include "decnode.h"

/**
   @brief To replace parallel array access.
 */
class ForestNode : public DecNode {
  static vector<double> splitQuant;

 public:

  /**
     @param[out] leafIdx outputs predictor index iff at leaf.
   */
  inline unsigned int advance(const double *rowT,
		       unsigned int &leafIdx) const {
    if (lhDel == 0) {
      leafIdx = predIdx;
      return 0;
    }
    else {
      return rowT[predIdx] <= splitVal.num ? lhDel : lhDel + 1;
    }
  }

  
  unsigned int advance(const class BVJagged *facSplit,
		       const unsigned int *rowT,
		       unsigned int tIdx,
		       unsigned int &leafIdx) const;
  

  unsigned int advance(const class FramePredict *framePredict,
		       const BVJagged *facSplit,
		       const unsigned int *rowFT,
		       const double *rowNT,
		       unsigned int tIdx,
		       unsigned int &leafIdx) const;



  static void Immutables(const vector<double> &feSplitQuant) {
    for (auto quant : feSplitQuant) {
      splitQuant.push_back(quant);
    }
  }

  
  static void DeImmutables() {
    splitQuant.clear();
  }
  
  
  void SplitUpdate(const class FrameTrain *frameTrain, const class BlockRanked *numRanked);
  

  /**
     @brief Initializer for tree node.
   */
  inline void Init() {
    predIdx = lhDel = 0;
    splitVal.num = 0.0;
  }


  inline void SetRank(const DecNode *decNode) {
    predIdx = decNode->predIdx;
    lhDel = decNode->lhDel;
    splitVal = decNode->splitVal;
    //    splitVal.rankRange = decNode->splitVal.rankRange;
  }


  /**
     @brief Copies decision node, converting offset to numeric value.

     @return void.
   */
  inline void SetOffset(const DecNode *decNode) {
    predIdx = decNode->predIdx;
    lhDel = decNode->lhDel;
    splitVal = decNode->splitVal;
    //    splitVal.num = decNode->splitVal.offset;
  }


  /**
     @brief Initializes leaf node.

     @return void.
   */
  inline void SetLeaf(unsigned int leafIdx) {
    predIdx = leafIdx;
    splitVal.num = 0.0;
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff lhDel value is nonzero.
   */
  inline bool Nonterminal() const {
    return lhDel != 0;
  }  


  inline unsigned int Pred() const {
    return predIdx;
  }


  inline unsigned int LHDel() const {
    return lhDel;
  }


  inline double Split() const {
    return splitVal.num;
  }

  
  /**
     @brief Multi-field accessor for a tree node.

     @param pred outputs the associated predictor index.

     @param lhDel outputs the increment to the LH descendant, if any.

     @param num outputs the numeric split value.

     @return void, with output reference parameters.
   */
  inline void RefNum(unsigned int &_pred,
		  unsigned int &_lhDel,
		  double &_num) const {
    _pred = predIdx;
    _lhDel = lhDel;
    _num = splitVal.num;
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const ForestNode *forestNode;
  const unsigned int nodeCount;
  const unsigned int *treeOrigin;
  const unsigned int nTree;
  class BVJagged *facSplit; // Consolidation of per-tree values.

  void NodeExport(vector<vector<unsigned int> > &predTree,
		  vector<vector<double> > &splitTree,
		  vector<vector<unsigned int> > &lhDelTree) const;

  
 public:

  Forest(const ForestNode _forestNode[],
	 unsigned int _nodeCount,
	 const unsigned int _origin[],
	 unsigned int _nTree,
	 unsigned int _facVec[],
	 size_t _facLen,
	 const unsigned int _facOrigin[],
	 unsigned int _nFac);

  ~Forest();

  /**
     @brief Accessor for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int NTree() const {
    return nTree;
  }


  inline const ForestNode *Node() const {
    return forestNode;
  }

  inline const unsigned int *Origin() const {
    return &treeOrigin[0];
  }

  inline const BVJagged *FacSplit() const {
    return facSplit;
  }
  
  /**
     @brief Determines height of individual tree height.

     @param tIdx is the tree index.

     @return Height of tree.
   */
  inline unsigned int TreeHeight(unsigned int tIdx) const {
    unsigned int heightInf = treeOrigin[tIdx];
    return tIdx < nTree - 1 ? treeOrigin[tIdx + 1] - heightInf : nodeCount - heightInf;
  }


  void Export(vector<vector<unsigned int> > &predTree,
	      vector<vector<double> > &splitTree,
	      vector<vector<unsigned int> > &lhDelTree,
	      vector<vector<unsigned int> > &facSplitTree) const;

};


/**
   @brief Class definition for crescent forest.
 */
class ForestTrain {
  vector<ForestNode> forestNode;
  vector<unsigned int> treeOrigin;
  vector<unsigned int> facOrigin;
  vector<unsigned int> facVec;


  /**
     @brief Maps tree-relative node index to forest-relative index.

     @param tIdx is the index of tree tree containing node.

     @param nodeOffset is the tree-relative index of the node.

     @return absolute index of node within forest.
   */
  inline unsigned int NodeIdx(unsigned int tIdx,
			      unsigned int nodeOffset) const {
    return treeOrigin[tIdx] + nodeOffset;
  }

  
  /**
     @return current accumulated size of crescent forest.
   */
  inline unsigned int Height() const {
    return forestNode.size();
  }


  /**
     @return current size of factor-valued splitting vector in crescent forest.
   */
  inline unsigned int SplitHeight() const {
    return facVec.size();
  }

  
 public:
  /**
     @brief Constructs block of trees for crescent forest, wrapping
     vectors allocated by the front-end bridge.

     @param[in] treeChunk is the number of trees to train.
   */
  ForestTrain(unsigned int treeChunk);

  
  ~ForestTrain();

  void BitProduce(const class BV *splitBits,
		  unsigned int bitEnd);

  void Origins(unsigned int tIdx);

  void Reserve(unsigned int nodeEst,
	       unsigned int facEst,
	       double slop);

  void NodeInit(unsigned int treeHeight);

  void SplitUpdate(const class FrameTrain *frameTrain,
		   const class BlockRanked *numRanked);

  void NonTerminal(const class FrameTrain *frameTrain,
		   unsigned int tIdx,
		   unsigned int idx,
		   const DecNode *decNode);

  
  /**
     @brief Derives number of trees from length of origin vector.

     @return number of trees.
   */
  inline unsigned int NTree() {
    return treeOrigin.size();
  }
  

  /**
     @return raw (byte) size of node region.
   */
  size_t NodeBytes() const {
    return Height() * sizeof(ForestNode);
  }

  /**
     @brief Outputs raw byes of node vector.

     @return void.
   */
  void NodeRaw(unsigned char rawOut[]) const {
    for (size_t i = 0; i < NodeBytes(); i++) {
      rawOut[i] = ((unsigned char *) &forestNode[0])[i];
    }
  }

  
  /**
     @return raw (byte) size of factor-splitting region.
   */
  size_t FacBytes() const {
    return SplitHeight() * sizeof(unsigned int);
  }
  

  /**
     @brief Outputs raw byes of node vector.

     @return void.
   */
  void FacRaw(unsigned char rawOut[]) const {
    for (size_t i = 0; i < FacBytes(); i++) {
      rawOut[i] = ((unsigned char *) &facVec[0])[i];
    }
  }


  /**
     @brief Exposes tree-origin vector.
   */
  const std::vector<unsigned int> &TreeOrigin() const {
    return treeOrigin;
  }


  /**
     @brief Exposes fac-origin vector.
   */
  const std::vector<unsigned int> &FacOrigin() const {
    return facOrigin;
  }
  

  /**
     @brief Sets looked-up nonterminal node to values passed.

     @return void.
  */
  inline void BranchProduce( unsigned int tIdx,
			   unsigned int nodeIdx,
			   const DecNode *decNode,
			   bool isFactor) {
    if (isFactor)
      forestNode[NodeIdx(tIdx, nodeIdx)].SetOffset(decNode);
    else
      forestNode[NodeIdx(tIdx, nodeIdx)].SetRank(decNode);
  }


  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  inline void LeafProduce(unsigned int tIdx,
			  unsigned int nodeIdx,
			  unsigned int leafIdx) {
    forestNode[NodeIdx(tIdx, nodeIdx)].SetLeaf(leafIdx);
  }
};

#endif
