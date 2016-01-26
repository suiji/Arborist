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
#include "callback.h"
#include "sample.h"
#include "rowrank.h"

//#include <iostream>
using namespace std;


/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @param yRanked is the sorted response.

   @param nRow is size of either input vector.

   @return void, with output reference vector.
 */
ResponseReg *Response::FactoryReg(const double yNum[], double yRanked[], unsigned int nRow) {
  return new ResponseReg(yNum, yRanked, nRow);
}


/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @return void.
*/
ResponseCtg *Response::FactoryCtg(const int feCtg[], const double feProxy[], unsigned int _nRow) {
  return new ResponseCtg(feCtg, feProxy, _nRow);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const int _yCtg[], const double _proxy[], unsigned int _nRow) : Response(_proxy) {
  yCtg = new unsigned int[_nRow];
  for (unsigned int i = 0; i < _nRow; i++) {
    yCtg[i] = _yCtg[i];
  }
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const double _y[]) : y(_y) {
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

   @param yRanked outputs the sorted response needed for quantile ranking.

   @param nRow is the length of the response vector.
 */
ResponseReg::ResponseReg(const double _y[], double yRanked[], unsigned int nRow) : Response(_y) {
  // The only client is quantile regression, via Sample::sample2Rank[],
  // but it is simpler to compute in all cases.
  //
  row2Rank = new unsigned int[nRow];
  unsigned int *rank2Row = new unsigned int[nRow];
  for (unsigned int i = 0; i < nRow; i++) {
    yRanked[i] = _y[i];
    rank2Row[i] = i;
  }

  // Can implement rank as match(_y, sort(_y)) in Rcpp.
  CallBack::QSortD(yRanked, (int *) rank2Row, 1, nRow);
  for (unsigned int rk = 0; rk < nRow; rk++) {
    unsigned int row = rank2Row[rk];
    row2Rank[row] = rk;
  }
  delete [] rank2Row;
}


/**
   @brief Regression subclass destructor.
*/
ResponseReg::~ResponseReg() {
  delete [] row2Rank;
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


/**
   @brief Destructor for classification.
 */
ResponseCtg::~ResponseCtg() {
  delete [] yCtg;
}
