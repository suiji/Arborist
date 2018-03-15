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
unique_ptr<ResponseCtg> Response::FactoryCtg(const unsigned int *feCtg,
					     const double *feProxy) {
  return make_unique<ResponseCtg>(feCtg, feProxy);
}


/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(const unsigned int *_yCtg,
			 const double *_proxy) :
  Response(_proxy), yCtg(_yCtg) {
}


/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(const double *_y) :
  y(_y) {
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
unique_ptr<ResponseReg> Response::FactoryReg(const double *yNum,
				  const unsigned int *_row2Rank) {
  return make_unique<ResponseReg>(yNum, _row2Rank);
}


/**
   @brief Regression subclass constructor.

   @param _y is the response vector.

 */
ResponseReg::ResponseReg(const double *_y,
			 const unsigned int *_row2Rank) :
  Response(_y),
  row2Rank(_row2Rank) {
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
  return Sample::FactoryCtg(Y(), rowRank, &yCtg[0]);
}
