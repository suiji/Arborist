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

#include "bv.h"
#include "response.h"
#include "sample.h"
#include "leaf.h"
#include "rowrank.h"
#include "index.h"
#include "pretree.h"

//#include <iostream>
using namespace std;


/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @return void.
*/
ResponseCtg *Response::FactoryCtg(const std::vector<unsigned int> &feCtg, const std::vector<double> &feProxy, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagRow> &bagRow, std::vector<double> &weight, unsigned int ctgWidth) {
  return new ResponseCtg(feCtg, feProxy, leafOrigin, leafNode, bagRow, weight, ctgWidth);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const std::vector<unsigned int> &_yCtg, const std::vector<double> &_proxy, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagRow> &bagRow, std::vector<double> &weight, unsigned int ctgWidth) : Response(_proxy, leafOrigin, leafNode, bagRow, weight, ctgWidth), yCtg(_yCtg) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagRow> &bagRow, std::vector<double> &weight, unsigned int ctgWidth) : y(_y), leaf(new LeafCtg(leafOrigin, leafNode, bagRow, weight, ctgWidth)) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagRow> &bagRow, std::vector<unsigned int> &rank) : y(_y), leaf(new LeafReg(leafOrigin, leafNode, bagRow, rank)) {
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

   @param yRanked is the sorted response.

   @return void, with output reference vector.
 */
ResponseReg *Response::FactoryReg(const std::vector<double> &yNum, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &bagRow, std::vector<unsigned int> &_rank) {
  return new ResponseReg(yNum, _row2Rank, _leafOrigin, _leafNode, bagRow, _rank);
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

   @param yRanked outputs the sorted response needed for quantile ranking.
 */
ResponseReg::ResponseReg(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<BagRow> &bagRow, std::vector<unsigned int> &rank) : Response(_y, leafOrigin, leafNode, bagRow, rank), row2Rank(_row2Rank) {
}


/**
   @brief Causes a block of classification trees to be sampled.

   @param rowRank is the predictor rank information.

   @param blockSize is the number of trees in the block.

   @return block of SampleCtg instances.
 */
PreTree **Response::BlockTree(const RowRank *rowRank, unsigned int blockSize) {
  sampleBlock = new Sample*[blockSize];
  for (unsigned int i = 0; i < blockSize; i++) {
    sampleBlock[i] = Sampler(rowRank);
  }

  return Index::BlockTrees(sampleBlock, blockSize);
}


/**
   @return Regression-style Sample object.
 */
Sample *ResponseReg::Sampler(const class RowRank *rowRank) {
  return Sample::FactoryReg(Y(), rowRank, row2Rank);
}


/**
   @return Classification-style Sample object.
 */
Sample *ResponseCtg::Sampler(const class RowRank *rowRank) {
  return Sample::FactoryCtg(Y(), rowRank, yCtg);
}


/**
   @brief Deletes Sample objects belonging to the current block.

   @param blockSize is the number of objects in the current block.

   @return void.
 */
void Response::DeBlock(unsigned int blockSize) {
  for (unsigned int blockIdx = 0; blockIdx < blockSize; blockIdx++) {
    delete sampleBlock[blockIdx];
  }
  delete [] sampleBlock;
  sampleBlock = 0;
}


/**
   @brief Fills in leaves for a tree.

   @param leafMap maps sampled indices to leaf indices.

   @param blockIdx is the block-based index of a Sample.

   @param tIdx is the absolute tree index.

   @return void, with side-effected Leaf object.
 */
void Response::Leaves(const std::vector<unsigned int> &leafMap, unsigned int blockIdx, unsigned int tIdx) {
  leaf->Leaves(sampleBlock[blockIdx], leafMap, tIdx);    
}


/**
   @brief Exposes in-bag vectors within sample block.

   @param blockIdx is the block-based index of a Sample.

   @return the in-bag vector for the tree at the referenced index.
 */
const BV *Response::TreeBag(unsigned int blockIdx) {
  return sampleBlock[blockIdx]->TreeBag();
}


/**
   @brief Initializes LeafCtg with estimated vector sizes.

   @param leafEst is the estimated number of leaves.

   @param bagEst is the estimated in-bag count.

   @return void, with side-effected leaf object.
*/
void Response::LeafReserve(unsigned int leafEst, unsigned int bagEst) {
  leaf->Reserve(leafEst, bagEst);
}
