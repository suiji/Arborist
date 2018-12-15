// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.h

   @brief Class definitions for sample-oriented aspects of training.

   @author Mark Seligman
 */

#ifndef ARBORIST_SAMPLE_H
#define ARBORIST_SAMPLE_H

#include <vector>
#include "typeparam.h"

#include "samplenux.h" // For now.


/**
   @brief Sum / count record for categorical indices.
 */
class SumCount {
  double sum;
  unsigned int sCount;

 public:
  void Init() {
    sum = 0.0;
    sCount = 0;
  }

  inline void Ref(double &_sum, unsigned int &_sCount) const {
    _sum = sum;
    _sCount = sCount;
  }
  
  
  inline void Accum(double _sum, unsigned int _sCount) {
    sum += _sum;
    sCount += _sCount;
  }


  /**
     @brief Subtracts contents of vector passed.
   */
  void decr(const SumCount &subtrahend) {
    sum -= subtrahend.sum;
    sCount -= subtrahend.sCount;
  }
};


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
  // Experimental coarse-grained control of locality:
  static const unsigned int locExp = 18;  // Log of locality threshold.

  /**
     @brief Maps an index into its bin.

     @param idx is the index in question.

     @return bin index.
   */
  static constexpr unsigned int binIdx(unsigned int idx) {
    return idx >> locExp;
  }

  
  //  vector<unsigned int> sample2Row;

 protected:
  static unsigned int nSamp;
  vector<SampleNux> sampleNode;
  vector<SumCount> ctgRoot;
  vector<unsigned int> row2Sample;
  unsigned int bagCount;
  double bagSum;

  static vector<unsigned int> rowSample(unsigned int nRow,
                                        unsigned int &bagCount_);

  /**
     @brief Bins a vector of indices for coarse locality.  Equivalent to
     the first pass of a radix sort.

     @param idx is an unordered vector of indices.

     @return binned version of index vector passed.
   */
  static vector<unsigned int> binIndices(const vector<unsigned int>& idx);


  /**
     @brief Tabulates a collection of indices by occurrence.

     @param idx is the vector of indices to be tabulated.

     @param sampleCount tabulates the occurrence count of each index.

     @return count of distinctly-sampled elements.
   */
  static unsigned int countSamples(vector<unsigned int>& idx,
                                   vector<unsigned int>& sampleCount);

  
  /**
     @brief Sets the stage, so to speak, for a newly-sampled response set.

     @param y is the proxy / response:  classification / summary.

     @param yCtg is true response / zero:  classification / regression.

     @param[out] treeBag records the bagged rows.

     @return void.
  */
  void preStage(const double y[],
                const unsigned int yCtg[],
                const class RowRank *rowRank,
                class BV *treeBag);

  virtual double addNode(double val, unsigned int sCount, unsigned int ctg) = 0;

 public:
  static class SampleCtg *FactoryCtg(const double y[],
                                     const class RowRank *rowRank,
                                     const unsigned int yCtg[],
                                     class BV *treeBag);

  static class SampleReg *FactoryReg(const double y[],
                                     const class RowRank *rowRank,
                                     const unsigned int *row2Rank,
                                     class BV *treeBag);
  
  virtual unique_ptr<class SplitNode> SplitNodeFactory(const class FrameTrain *frameTrain, const RowRank *rowRank) const = 0;

  static void Immutables(unsigned int nSamp_);
  static void DeImmutables();
  Sample();
  virtual ~Sample();

  unique_ptr<class SamplePred> stage(const class RowRank *rowRank,
             vector<class StageCount> &stageCount);

  /**
     @brief Accessor for root category census.
   */
  inline const vector<SumCount> &CtgRoot() const {
    return ctgRoot;
  }

  
  /**
     @brief Accessor for sample count.
   */
  static inline unsigned int getNSamp() {
    return nSamp;
  }


  /**
     @brief Accessor for bag count.
   */
  inline unsigned int getBagCount() const {
    return bagCount;
  }

  
  inline double  getBagSum() const {
    return bagSum;
  }

  
  inline unsigned int getSCount(unsigned int sIdx) const {
    return sampleNode[sIdx].getSCount();
  }


  /**
     @brief Determines whether a given row is sampled.

     @param row is the row number in question.

     @param[out] sIdx is the (possibly default) sample index for row.

     @return true iff row is sampled.
   */
  inline bool sampledRow(unsigned int row, unsigned int &sIdx) const {
    sIdx = row2Sample[row];
    return sIdx < bagCount;
  }

  /**
     @brief References leaf-specific fields.
   */
  inline void refLeaf(unsigned int sIdx, FltVal &_sum, unsigned int &_ctg) const {
    _ctg = sampleNode[sIdx].refLeaf(_sum);
  }

  
  inline FltVal getSum(int sIdx) const {
    return sampleNode[sIdx].getSum();
  }
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  unsigned int *sample2Rank; // Only client currently leaf-based methods.

  /**
   @brief Compresses row->rank map to sIdx->rank.  Requires
   that row2Sample[] be complete:  PreStage().

   @param row2Rank[] is the response ranking, by row.

   @return void, with side-effected sample2Rank[].
  */
  void setRank(const unsigned int *row2Rank);

 public:
  SampleReg();
  ~SampleReg();
  unique_ptr<class SplitNode> SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const;

  
  inline double addNode(double yVal,
                        unsigned int sCount,
                        unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount);
    sampleNode.emplace_back(sNode);

    return ySum;
  }
  

  /**
     @brief Looks up the rank of the response at a given sample index.

     @param sIdx is a sample index.

     @return rank of outcome.
   */
  inline unsigned int getRank(unsigned int sIdx) const {
    return sample2Rank[sIdx];
  }



  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.

     @param row2Rank is the response ranking, by row.

     @param[out] treeBag encodes the bagged rows for the tree.

     @return void.
  */
  void preStage(const double y[], const unsigned int *row2Rank, const class RowRank *rowRank, class BV *treeBag);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {

 public:
  SampleCtg();
  ~SampleCtg();

  unique_ptr<class SplitNode> SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const;

  inline double addNode(double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount, ctg);
    sampleNode.emplace_back(sNode);
    ctgRoot[ctg].Accum(ySum, sCount);

    return ySum;
  }
  
  
  /**
     @brief Samples the response, sets in-bag bits and stages.

     @param yCtg is the response vector.

     @param y is the proxy response vector.

     @param rowRank .

     @param[out] treeBag records the bagged rows.

     @return void, with output vector parameter.
  */
  void preStage(const unsigned int yCtg[],
                const double y[],
                const class RowRank *rowRank,
                class BV *treeBag);
};


#endif
