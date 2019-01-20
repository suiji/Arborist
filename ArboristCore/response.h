// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file response.h

   @brief Class definitions for representing response-specific aspects of training, especially regression versus categorical support.

   @author Mark Seligman

 */


#ifndef ARBORIST_RESPONSE_H
#define ARBORIST_RESPONSE_H

#include <vector>

#include "typeparam.h"

/**
   @brief Methods and members for management of response-related computations.
 */
class Response {
  const double *y;

 public:
  Response(const double *_y);
  virtual ~Response();

  inline const double *Y() const {
    return y;
  }


  static unique_ptr<class ResponseReg> factoryReg(const double* yNum,
					    const unsigned int* row2Rank_);

  static unique_ptr<class ResponseCtg> factoryCtg(const unsigned int* feCtg,
                                                  const double* feProxy);

  virtual shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                              class BV* treeBag) const = 0;
};


/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
  const unsigned int* row2Rank; // Facilitates rank[] output.
 public:

  ResponseReg(const double* y_,
	      const unsigned int* row2Rank_);

  ~ResponseReg();
  shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                      class BV* treeBag) const;

};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  const unsigned int* yCtg; // 0-based factor-valued response.
 public:

  ResponseCtg(const unsigned int *yCtg_,
	      const double *proxy_);

  ~ResponseCtg();
  shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                      class BV* treeBag) const;
};

#endif
