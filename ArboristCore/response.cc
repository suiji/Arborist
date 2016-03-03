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
#include "sample.h"
#include "rowrank.h"

//#include <iostream>
using namespace std;


/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @param yRanked is the sorted response.

   @return void, with output reference vector.
 */
ResponseReg *Response::FactoryReg(const std::vector<double> &yNum, const std::vector<unsigned int> &_row2Rank) {
  return new ResponseReg(yNum, _row2Rank);
}


/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @return void.
*/
ResponseCtg *Response::FactoryCtg(const std::vector<unsigned int> &feCtg, const std::vector<double> &feProxy) {
  return new ResponseCtg(feCtg, feProxy);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const std::vector<unsigned int> &_yCtg, const std::vector<double> &_proxy) : Response(_proxy), yCtg(_yCtg) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const std::vector<double> &_y) : y(_y) {
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

   @param yRanked outputs the sorted response needed for quantile ranking.
 */
ResponseReg::ResponseReg(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank) : Response(_y), row2Rank(_row2Rank) {
}


/**
   @brief Causes a block of classification trees to be sampled.

   @param rowRank is the predictor rank information.

   @param tCount is the number of trees in the block.

   @return block of SampleCtg instances.
 */
SampleCtg **ResponseCtg::BlockSample(const RowRank *rowRank, int tCount) {
  SampleCtg **sampleBlock = new SampleCtg*[tCount];
  for (int i = 0; i < tCount; i++) {
    sampleBlock[i] = SampleCtg::Factory(y, rowRank, yCtg);
  }

  return sampleBlock;
}


/**
   @brief Causes a block of regression trees to be sampled.

   @param rowRank is the predictor rank information.

   @param tCount is the number of trees in the block.

   @return block of SampleReg instances.
 */
SampleReg **ResponseReg::BlockSample(const RowRank *rowRank, int tCount) {
  SampleReg **sampleBlock = new SampleReg*[tCount];
  for (int i = 0; i < tCount; i++) {
    sampleBlock[i] = SampleReg::Factory(y, rowRank, row2Rank);
  }

  return sampleBlock;
}
