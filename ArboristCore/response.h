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

/**
   @brief Methods and members for management of response-related computations.
 */
class Response {
 public:
  const double *y;
  Response(const double _y[]);
  static class ResponseReg *FactoryReg(const double yNum[], double yRanked[], unsigned int nRow);
  static class ResponseCtg *FactoryCtg(const int feCtg[], const double feProxy[]);
  
  virtual ~Response(){}
};

/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
  class SampleReg* SampleRows(const class PredOrd *predOrd);

 public:
  ResponseReg(const double _y[], double yRanked[], unsigned int nRow);
  ~ResponseReg();
  class SampleReg **BlockSample(const class PredOrd *predOrd, int tCount);
  unsigned int *row2Rank;
};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  class SampleCtg* SampleRows(const class PredOrd *predOrd);
 public:
  const int *yCtg; // The original factor-valued response.

  class SampleCtg **BlockSample(const class PredOrd *predOrd, int tCount);
  ResponseCtg(const int _yCtg[], const double _yProxy[]);
  ~ResponseCtg();
  
  static int CtgSum(unsigned int sIdx, double &sum);
};

#endif
