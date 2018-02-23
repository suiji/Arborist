// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman
 */


#include "bv.h"
#include "forest.h"
#include "frameblock.h"
#include "rowrank.h"
#include "predict.h"


vector<double> ForestNode::splitQuant;


/**
   @brief Crescent constructor for training.
*/
ForestTrain::ForestTrain(unsigned int nTree) :
  forestNode(vector<ForestNode>(0)),
  treeOrigin(vector<unsigned int>(nTree)),
  facOrigin(vector<unsigned int>(nTree)),
  facVec(vector<unsigned int>(0)) {
}


ForestTrain::~ForestTrain() {
}


/**
   @brief Constructor for prediction.
*/
Forest::Forest(const ForestNode _forestNode[],
	       unsigned int _nodeCount,
	       const unsigned int _origin[],
	       unsigned int _nTree,
	       unsigned int _facVec[],
	       size_t _facLen,
	       const unsigned int _facOrigin[],
	       unsigned int _nFac) :
  forestNode(_forestNode),
  nodeCount(_nodeCount),
  treeOrigin(_origin),
  nTree(_nTree),
  facSplit(new BVJagged(_facVec, _facLen, _facOrigin, _nFac)) {
}


/**
 */ 
Forest::~Forest() {
  delete facSplit;
}


/**
   @brief Dispatches prediction method based on available predictor types.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Forest::PredictAcross(Predict *predict,
			   unsigned int rowStart,
			   unsigned int rowEnd,
			   const class BitMatrix *bag) const {
  if (predict->PredMap()->NPredFac() == 0)
    PredictAcrossNum(predict, rowStart, rowEnd, bag);
  else if (predict->PredMap()->NPredNum() == 0)
    PredictAcrossFac(predict, rowStart, rowEnd, bag);
  else
    PredictAcrossMixed(predict, rowStart, rowEnd, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(Predict *predict,
			      unsigned int rowStart,
			      unsigned int rowEnd,
			      const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      PredictRowNum(predict, row, predict->RowNum(row - rowStart), row - rowStart, bag);
    }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossFac(Predict *predict,
			      unsigned int rowStart,
			      unsigned int rowEnd,
			      const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      PredictRowFac(predict, row, predict->RowFac(row - rowStart), row - rowStart, bag);
  }
  }

}


/**
   @brief Multi-row prediction with predictors of both numeric and factor type.

   @param rowStart is the first row in the block.

   @param rowEnd is the first row beyond the block.

   @param bag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(Predict *predict,
				unsigned int rowStart,
				unsigned int rowEnd,
				const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      PredictRowMixed(predict, row, predict->RowNum(row - rowStart), predict->RowFac(row - rowStart), row - rowStart, bag);
    }
  }

}


/**
   @brief Prediction with predictors of only numeric type.

   @param row is the row of data over which a prediction is made.

   @param rowT is a numeric data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */

void Forest::PredictRowNum(Predict *predict,
			   unsigned int row,
			   const double rowT[],
			   unsigned int blockRow,
			   const class BitMatrix *bag) const {
  unsigned int noLeaf = predict->NoLeaf();
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(rowT, leafIdx);
    }

    predict->LeafIdx(blockRow, tIdx, leafIdx);
  }
}


/**
   @brief Prediction with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Forest::PredictRowFac(Predict *predict,
			   unsigned int row,
			   const unsigned int rowT[],
			   unsigned int blockRow,
			   const class BitMatrix *bag) const {
  unsigned int noLeaf = predict->NoLeaf();
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(facSplit, rowT, tIdx, leafIdx);
    }

    predict->LeafIdx(blockRow, tIdx, leafIdx);
  }
}


unsigned int ForestNode::Advance(const BVJagged *facSplit, const unsigned int rowT[], unsigned int tIdx, unsigned int &leafIdx) const {
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    unsigned int bitOff = splitVal.offset + rowT[predIdx];
    return facSplit->TestBit(tIdx, bitOff) ? lhDel : lhDel + 1;
  }
}


/**
   @brief Prediction with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Forest::PredictRowMixed(Predict *predict,
			     unsigned int row,
			     const double rowNT[],
			     const unsigned int rowFT[],
			     unsigned int blockRow,
			     const class BitMatrix *bag) const {
  unsigned int noLeaf = predict->NoLeaf();
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int leafIdx = noLeaf;
    while (leafIdx == noLeaf) {
      idx += forestNode[idx].Advance(predict->PredMap(), facSplit, rowFT, rowNT, tIdx, leafIdx);
    }

    predict->LeafIdx(blockRow, tIdx, leafIdx);
  }
}


unsigned int ForestNode::Advance(const FramePredict *framePredict,
				 const BVJagged *facSplit,
				 const unsigned int *rowFT,
				 const double *rowNT,
				 unsigned int tIdx,
				 unsigned int &leafIdx) const {
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    bool isFactor;
    unsigned int blockIdx = framePredict->FacIdx(predIdx, isFactor);
    return isFactor ? (facSplit->TestBit(tIdx, splitVal.offset + rowFT[blockIdx]) ? lhDel : lhDel + 1) : (rowNT[blockIdx] <= splitVal.num ? lhDel : lhDel + 1);
  }
}


/**
 */
void ForestTrain::NodeInit(unsigned int treeHeight) {
  ForestNode fn;
  fn.Init();
  forestNode.insert(forestNode.end(), treeHeight, fn);
}


/**
   @brief Produces new splits for an entire tree.
 */
void ForestTrain::BitProduce(const BV *splitBits,
			     unsigned int bitEnd) {
  splitBits->Consume(facVec, bitEnd);
}


/**
  @brief Reserves space in the relevant vectors for new trees.
 */
void ForestTrain::Reserve(unsigned int blockHeight,
			  unsigned int blockFac,
			  double slop) {
  forestNode.reserve(slop * blockHeight);
  if (blockFac > 0) {
    facVec.reserve(slop * blockFac);
  }
}


/**
   @brief Registers current vector sizes of crescent forest as origin values.

   @param tIdx is current tree index.
   
   @return void.
 */
void ForestTrain::Origins(unsigned int tIdx) {
  treeOrigin[tIdx] = Height();
  facOrigin[tIdx] = SplitHeight();
}


void ForestTrain::NonTerminal(const FrameTrain *frameTrain,
			      unsigned int tIdx,
			      unsigned int idx,
			      const DecNode *decNode) {
  BranchProduce(tIdx, idx, decNode, frameTrain->IsFactor(decNode->predIdx));
}


/**
   @brief Post-pass to update numerical splitting values from ranks.

   @param rowRank holds the presorted predictor values.

   @return void
 */
void ForestTrain::SplitUpdate(const FrameTrain *frameTrain,
			      const RowRank *rowRank) {
  for (auto & fn : forestNode) {
    fn.SplitUpdate(frameTrain, rowRank);
  }
}


/**
   @brief Assigns value at quantile rank to numerical split.

   @param rowRank holds the presorted predictor values.

   @return void.
 */
void ForestNode::SplitUpdate(const FrameTrain *frameTrain,
			     const RowRank *rowRank) {
  if (Nonterminal() && !frameTrain->IsFactor(predIdx)) {
    splitVal.num = rowRank->QuantRank(predIdx, splitVal.rankRange, splitQuant);
  }
}


void Forest::Export(vector<vector<unsigned int> > &predTree,
		    vector<vector<double> > &splitTree,
		    vector<vector<unsigned int> > &lhDelTree,
		    vector<vector<unsigned int> > &facSplitTree) const {
  NodeExport(predTree, splitTree, lhDelTree);
  facSplit->Export(facSplitTree);
}


/**
   @brief Unpacks node fields into vector of per-tree vectors.

   @return void, with output reference vectors.
 */
void Forest::NodeExport(vector<vector<unsigned int> > &pred,
			vector<vector<double> > &split,
			vector<vector<unsigned int> > &lhDel) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int nodeIdx = 0; nodeIdx < TreeHeight(tIdx); nodeIdx++) {
      pred[tIdx].push_back(forestNode[nodeIdx].Pred());
      lhDel[tIdx].push_back(forestNode[nodeIdx].LHDel());

      // Not quite:  must distinguish numeric from bit-packed:
      split[tIdx].push_back(forestNode[nodeIdx].Split());
    }
  }
}
