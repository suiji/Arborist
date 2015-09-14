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

//#include <iostream>
using namespace std;


/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @return void.
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
ResponseCtg *Response::FactoryCtg(const int feCtg[], const double feProxy[]) {
  return new ResponseCtg(feCtg, feProxy);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const int _yCtg[], const double _proxy[]) : Response(_proxy), yCtg(_yCtg) {
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
  CallBack::QSortD(yRanked, rank2Row, 1, nRow);
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


SampleCtg **ResponseCtg::BlockSample(const PredOrd *predOrd, int tCount) {
  SampleCtg **sampleBlock = new SampleCtg*[tCount];
  for (int i = 0; i < tCount; i++) {
    sampleBlock[i] = SampleRows(predOrd);
  }

  return sampleBlock;
}


SampleReg **ResponseReg::BlockSample(const PredOrd *predOrd, int tCount) {
  SampleReg **sampleBlock = new SampleReg*[tCount];
  for (int i = 0; i < tCount; i++) {
    sampleBlock[i] = SampleRows(predOrd);
  }

  return sampleBlock;
}
 
 
/**
   @brief Row-sampling entry for regression tree.

   @param predOrd is the ordered predictor table.

   @return void, with output parameter vectors.
*/
SampleReg *ResponseReg::SampleRows(const PredOrd *predOrd) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->Stage(y, row2Rank, predOrd);

  return sampleReg;
}


/**
   @brief Row-sampling entry for classification tree.

   @param predOrd is the ordered predictor table.

   @return Sample object, plus output param.
*/
SampleCtg *ResponseCtg::SampleRows(const PredOrd *predOrd) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->Stage(yCtg, y, predOrd);

  return sampleCtg;
}


/**
   @brief Destructor for categorical response.
 */
ResponseCtg::~ResponseCtg() {
}
