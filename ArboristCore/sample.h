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
  //  vector<unsigned int> sample2Row;

 protected:
  static unsigned int nSamp;
  vector<SampleNux> sampleNode;
  vector<SumCount> ctgRoot;
  vector<unsigned int> row2Sample;
  unsigned int bagCount;
  double bagSum;

  static unsigned int rowSample(vector<unsigned int> &sCountRow);
  
  
  void PreStage(const double y[],
                const unsigned int yCtg[],
                const class RowRank *rowRank,
                class BV *treeBag);

  virtual double setNode(unsigned int sIdx, double val, unsigned int sCount, unsigned int ctg) = 0;

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

  unique_ptr<class SamplePred> Stage(const class RowRank *rowRank,
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
  void SetRank(const unsigned int *row2Rank);

 public:
  SampleReg();
  ~SampleReg();
  unique_ptr<class SplitNode> SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const;

  
  inline double setNode(unsigned int sIdx,
                        double yVal,
                        unsigned int sCount,
                        unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount);
    sampleNode[sIdx] = sNode;

    return ySum;
  }
  
  
  inline unsigned int Rank(unsigned int sIdx) const {
    return sample2Rank[sIdx];
  }


  void PreStage(const double y[], const unsigned int *row2Rank, const class RowRank *rowRank, class BV *treeBag);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {

 public:
  SampleCtg();
  ~SampleCtg();

  unique_ptr<class SplitNode> SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const;

  inline double setNode(unsigned int sIdx, double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.init(yVal, sCount, ctg);
    sampleNode[sIdx] = sNode;
    ctgRoot[ctg].Accum(ySum, sCount);

    return ySum;
  }
  
  
  void PreStage(const unsigned int yCtg[], const double y[], const class RowRank *rowRank, class BV *treeBag);
};


#endif
