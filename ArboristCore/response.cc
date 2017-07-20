// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file response.cc

   @brief Methods maintaining the response-specific aspects of training.

   @author Mark Seligman

 */

#include "response.h"
#include "predblock.h"
#include "sample.h"
#include "leaf.h"
#include "rowrank.h"

//#include <iostream>
//using namespace std;


/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @return void.
*/
ResponseCtg *Response::FactoryCtg(const std::vector<unsigned int> &feCtg, const std::vector<double> &feProxy, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth) {
  return new ResponseCtg(feCtg, feProxy, leafOrigin, leafNode, bagLeaf, bagBits, weight, ctgWidth);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const std::vector<unsigned int> &_yCtg, const std::vector<double> &_proxy, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth) : Response(_proxy, leafOrigin, leafNode, bagLeaf, bagBits, weight, ctgWidth), nCtg(ctgWidth), yCtg(_yCtg) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth) : y(_y), leaf(new LeafCtg(leafOrigin, leafNode, bagLeaf, bagBits, y.size(), weight, ctgWidth)) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits) : y(_y), leaf(new LeafReg(leafOrigin, leafNode, bagLeaf, bagBits, y.size())) {
}


Response::~Response() {
  delete leaf;
}


ResponseCtg::~ResponseCtg() {
}


ResponseReg::~ResponseReg() {
}


/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @return void, with output reference vector.
 */
ResponseReg *Response::FactoryReg(const std::vector<double> &yNum, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits) {
  return new ResponseReg(yNum, _row2Rank, _leafOrigin, _leafNode, bagLeaf, bagBits);
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

 */
ResponseReg::ResponseReg(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits) : Response(_y, leafOrigin, leafNode, bagLeaf, bagBits), row2Rank(_row2Rank) {
}


/**
   @brief Causes a block trees to be sampled and trained.

   @param rowRank is the predictor rank information.

   @param sampleBlock summarizes a block of sampled rows.

   @return void.
 */
void Response::TreeBlock(const RowRank *rowRank, std::vector<Sample*> &sampleBlock) const {
  for (auto & sample : sampleBlock) {
    sample = RootSample(rowRank);
  }
}


/**
   @return Regression-style Sample object.
 */
Sample *ResponseReg::RootSample(const RowRank *rowRank) const {
  return Sample::FactoryReg(Y(), rowRank, row2Rank);
}


/**
   @return Classification-style Sample object.
 */
Sample *ResponseCtg::RootSample(const RowRank *rowRank) const {
  return Sample::FactoryCtg(Y(), rowRank, yCtg, nCtg);
}


/**
   @brief Fills in leaves for a tree using current Sample.

   @param leafMap maps sampled indices to leaf indices.

   @param tIdx is the absolute tree index.

   @return void, with side-effected Leaf object.
 */
void Response::Leaves(const PMTrain *pmTrain, const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) const {
  leaf->Leaves(pmTrain, sample, leafMap, tIdx);
}


/**
   @brief Initializes LeafCtg with estimated vector sizes.

   @param leafEst is the estimated number of leaves.

   @param bagEst is the estimated in-bag count.

   @return void, with side-effected leaf object.
*/
void Response::LeafReserve(unsigned int leafEst, unsigned int bagEst) const {
  leaf->Reserve(leafEst, bagEst);
}
