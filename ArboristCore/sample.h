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

  
 protected:
  const class RowRank* rowRank;
  
  static unsigned int nSamp;
  vector<SampleNux> sampleNode;
  vector<SumCount> ctgRoot;
  vector<unsigned int> row2Sample; // Maps row index to sample index.
  unsigned int bagCount; // Number of distinct bagged (sampled) rows.
  double bagSum; // Sum of bagged responses.


  /**
     @brief Samples and counts occurrences of each sampled row
     index.

     @param[out] sCountRow outputs the number of times a given row is
     sampled.

     @return count of bagged rows.
  */
  static unsigned int rowSample(vector<unsigned int> &sCountRow);

  
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
     @brief Samples rows and counts resulting occurrences.

     @param y is the proxy / response:  classification / summary.

     @param yCtg is true response / zero:  classification / regression.

     @param[out] treeBag records the bagged rows as high bits.
  */
  void bagSamples(const double y[],
                const unsigned int yCtg[],
                class BV *treeBag);

  virtual double addNode(double val, unsigned int sCount, unsigned int ctg) = 0;

 public:

  /**
     @brief Static entry for discrete response.

     @param y is a real-valued proxy for the training response.

     @param rowRank summarizes the ranked observations.

     @param yCtg is the training response.

     @param[out] treeBag outputs bit-encoded indicator of sampled rows.

     @return new SampleCtg instance.
   */
  static shared_ptr<class SampleCtg> factoryCtg(const double y[],
                                                const class RowRank *rowRank,
                                                const unsigned int yCtg[],
                                                class BV *treeBag);

  /**
     @brief Static entry for continuous response.

     @param y is the training response.

     @param rowRank summarizes the ranked observations.

     @param[out] treeBag outputs bit-encoded indicator of sampled rows.

     @return new SampleReg instance.
   */
  static shared_ptr<class SampleReg>factoryReg(const double y[],
                                               const class RowRank *rowRank,
                                               class BV *treeBag);
  
  virtual unique_ptr<class SplitNode> splitNodeFactory(const class FrameTrain *frameTrain) const = 0;

  /**
     @brief Lights off static initializations needed for sampling.

     @param nSamp_ is the number of samples.
  */
  static void immutables(unsigned int nSamp_);


  /**
     @brief Resets statics.
  */
  static void deImmutables();


  /**
     @brief Constructor.

     @param rowRank summarizes predictor ranks by row.
   */
  Sample(const class RowRank* rowRank_);


  virtual ~Sample();


  /**
     @brief

     @return array of joined sample/predictor records.
  */
  unique_ptr<class SamplePred> predictors() const;
  

  /**
     @brief Invokes RowRank staging methods and caches compression map.
  */
  vector<class StageCount> stage(class SamplePred* samplePred) const;


  /**
     @brief Getter for root category census vector.
   */
  inline const vector<SumCount> &getCtgRoot() const {
    return ctgRoot;
  }

  
  /**
     @brief Accessor for sample count.
   */
  static inline unsigned int getNSamp() {
    return nSamp;
  }


  /**
     @brief Getter for bag count.
   */
  inline unsigned int getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for sum of bagged responses.
   */
  inline double  getBagSum() const {
    return bagSum;
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
     @brief Accumulates 'sum' field into various containers.

     @param sIdx is the sample index.

     @param[in, out] bulkSum accumulates sums irrespective of category.

     @param[in, out] ctgSum accumulates sums by category.
   */
  inline void accum(unsigned int sIdx,
                    double &bulkSum,
                    double *ctgSum) const {
    unsigned int ctg;
    FltVal sum = sampleNode[sIdx].refCtg(ctg);
    bulkSum += sum;
    ctgSum[ctg] += sum;
  }


  /**
     @brief Getter for sample count.

     @param sIdx is the sample index.
   */
  inline unsigned int getSCount(unsigned int sIdx) const {
    return sampleNode[sIdx].getSCount();
  }


  /**
     @brief Getter for the sampled response sum.

     @param sIdx is the sample index.
   */
  inline FltVal getSum(int sIdx) const {
    return sampleNode[sIdx].getSum();
  }
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {

 public:
  SampleReg(const class RowRank* rowRank_);
  ~SampleReg();
  unique_ptr<class SplitNode> splitNodeFactory(const FrameTrain *frameTrain) const;

  
  inline double addNode(double yVal,
                        unsigned int sCount,
                        unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount);
    sampleNode.emplace_back(sNode);

    return ySum;
  }
  

  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.

     @param[out] treeBag encodes the bagged rows for the tree.
  */
  void bagSamples(const double y[], class BV *treeBag);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {

 public:
  SampleCtg(const class RowRank* rowRank_);
  ~SampleCtg();

  unique_ptr<class SplitNode> splitNodeFactory(const FrameTrain *frameTrain) const;

  /**
     @brief Appends a sample summary to the node vector.

     @return sum of sampled response values.
   */
  inline double addNode(double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount, ctg);
    sampleNode.emplace_back(sNode);
    ctgRoot[ctg].Accum(ySum, sCount);

    return ySum;
  }
  
  
  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.

     @param[out] treeBag records the bagged rows.
  */
  void bagSamples(const unsigned int yCtg[],
                const double y[],
                class BV *treeBag);
};


#endif
