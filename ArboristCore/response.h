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

/**
   @brief Methods and members for management of response-related computations.
 */
class Response {
 protected:
  const std::vector<double> &y;
 public:
  Response(const std::vector<double> &_y);
  static class ResponseReg *FactoryReg(const std::vector<double> &yNum, const std::vector<unsigned int> &_row2Rank);
  static class ResponseCtg *FactoryCtg(const std::vector<unsigned int> &feCtg, const std::vector<double> &feProxy);
};

/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
  class SampleReg* SampleRows(const class RowRank *rowRank);
  const std::vector<unsigned int> &row2Rank; // Facilitates rank[] output.

 public:
  ResponseReg(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank);
  class SampleReg **BlockSample(const class RowRank *rowRank, int tCount);
};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  class SampleCtg* SampleRows(const class RowRank *rowRank);
  const std::vector<unsigned int> &yCtg; // 0-based factor-valued response.
 public:

  class SampleCtg **BlockSample(const class RowRank *rowRank, int tCount);
  ResponseCtg(const std::vector<unsigned int> &_yCtg, const std::vector<double> &yProxy);
  
  static int CtgSum(unsigned int sIdx, double &sum);
};

#endif
