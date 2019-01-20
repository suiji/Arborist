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
#include "framemap.h"
#include "sample.h"
#include "rowrank.h"


/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @return void.
*/
unique_ptr<ResponseCtg> Response::factoryCtg(const unsigned int* feCtg,
					     const double* feProxy) {
  return make_unique<ResponseCtg>(feCtg, feProxy);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const unsigned int* yCtg_,
			 const double* proxy_) :
  Response(proxy_), yCtg(yCtg_) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const double *y_) :
  y(y_) {
}


Response::~Response() {
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
unique_ptr<ResponseReg> Response::factoryReg(const double* yNum,
				  const unsigned int* row2Rank_) {
  return make_unique<ResponseReg>(yNum, row2Rank_);
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

 */
ResponseReg::ResponseReg(const double* y_,
			 const unsigned int* row2Rank_) :
  Response(y_),
  row2Rank(row2Rank_) {
}


/**
   @return Regression-style Sample object.
 */
shared_ptr<Sample> ResponseReg::rootSample(const RowRank* rowRank,
                                BV* treeBag) const {
  return Sample::factoryReg(Y(), rowRank, row2Rank, treeBag);
}


/**
   @return Classification-style Sample object.
 */
shared_ptr<Sample> ResponseCtg::rootSample(const RowRank* rowRank,
                                BV* treeBag) const {
  return Sample::factoryCtg(Y(), rowRank, &yCtg[0], treeBag);
}
