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
  const std::vector<double> &y;
  class Leaf *leaf;

 public:
  Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth);
  Response(const std::vector<double> &_y, std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits);
  virtual ~Response();

  inline const std::vector<double> &Y() const {
    return y;
  }


  static class ResponseReg *FactoryReg(const std::vector<double> &yNum, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits);
  static class ResponseCtg *FactoryCtg(const std::vector<unsigned int> &feCtg, const std::vector<double> &feProxy, std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth);

  void TreeBlock(const class RowRank *rowRank, std::vector<class Sample*> &sampleBlock) const;
  void LeafReserve(unsigned int leafEst, unsigned int bagEst) const;
  void Leaves(const class PMTrain *pmTrain, const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) const;

  virtual class Sample* RootSample(const class RowRank *rowRank) const = 0;
};


/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
  const std::vector<unsigned int> &row2Rank; // Facilitates rank[] output.
 public:

  ResponseReg(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits);
  ~ResponseReg();
  class Sample *RootSample(const class RowRank *rowRank) const;
};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  const unsigned int nCtg;
  const std::vector<unsigned int> &yCtg; // 0-based factor-valued response.
 public:

  ResponseCtg(const std::vector<unsigned int> &_yCtg, const std::vector<double> &_proxy, std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, std::vector<class BagLeaf> &bagLeaf, std::vector<unsigned int> &bagBits, std::vector<double> &weight, unsigned int ctgWidth);
  ~ResponseCtg();
  class Sample *RootSample(const class RowRank *rowRank) const;
};

#endif
