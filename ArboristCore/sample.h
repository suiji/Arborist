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
#include "param.h"

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
  void Decr(const SumCount &subtrahend) {
    sum -= subtrahend.sum;
    sCount -= subtrahend.sCount;
  }
};


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
  class BV *treeBag;
  std::vector<unsigned int> sample2Row;

 protected:
  static unsigned int nSamp;
  std::vector<SampleNux> sampleNode;
  std::vector<SumCount> ctgRoot;
  unsigned int bagCount;
  double bagSum;

  static unsigned int RowSample(std::vector<unsigned int> &sCountRow);
  
  
  void PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const class RowRank *rowRank, std::vector<unsigned int> &row2Sample);

  virtual double SetNode(unsigned int sIdx, double val, unsigned int sCount, unsigned int ctg) = 0;

 public:
  static class SampleCtg *FactoryCtg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &yCtg, unsigned int _nCtg, std::vector<unsigned int> &row2Sample);
  static class SampleReg *FactoryReg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &row2Rank, std::vector<unsigned int> &row2Sample);
  virtual class SplitPred *SplitPredFactory(const class PMTrain *pmTrain, const RowRank *rowRank) const = 0;

  static void Immutables(unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool _withRepl);
  static void DeImmutables();
  Sample(unsigned int _nRow, unsigned int nCtg);
  virtual ~Sample();

  static void StageFactory(const class PMTrain *pmTrain, const class RowRank *rowRank, const class Response *reponse, Sample *&sample, class SplitPred *&splitPred, class SamplePred *&samplePred, std::vector<class StageCount> &stageCount);

  void Stage(const class RowRank *rowRank, const std::vector<unsigned int> &row2Sample, class SamplePred *&samplePred, std::vector<StageCount> &stageCount);
  void RowInvert(const std::vector<unsigned int> &row2Sample);


  /**
     @brief Accessor for root category census.
   */
  inline const std::vector<SumCount> &CtgRoot() const {
    return ctgRoot;
  }

  
  /**
     @brief Accessor for sample count.
   */
  static inline unsigned int NSamp() {
    return nSamp;
  }


  /**
   */
  unsigned int Sample2Row(unsigned int sIdx) const {
    return sample2Row[sIdx];
  }
  

  /**
     @brief Accessor for bag count.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }

  
  inline double BagSum() const {
    return bagSum;
  }

  
  inline const class BV *TreeBag() const {
    return treeBag;
  }


  inline unsigned int SCount(unsigned int sIdx) const {
    return sampleNode[sIdx].SCount();
  }


  /**
     @brief References leaf-specific fields.
   */
  inline void RefLeaf(unsigned int sIdx, FltVal &_sum, unsigned int &_ctg) const {
    _ctg = sampleNode[sIdx].RefLeaf(_sum);
  }

  
  inline FltVal Sum(int sIdx) const {
    return sampleNode[sIdx].Sum();
  }
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  unsigned int *sample2Rank; // Only client currently leaf-based methods.
  void SetRank(const std::vector<unsigned int> &row2Sample, const std::vector<unsigned int> &row2Rank);

 public:
  SampleReg(unsigned int _nRow);
  ~SampleReg();
  SplitPred *SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank) const;

  
  inline double SetNode(unsigned int sIdx, double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.Init(yVal, sCount);
    sampleNode[sIdx] = sNode;

    return ySum;
  }
  
  
  inline unsigned int Rank(unsigned int sIdx) const {
    return sample2Rank[sIdx];
  }


  void PreStage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const class RowRank *rowRank, std::vector<unsigned int> &row2Sample);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {
  const unsigned int nCtg;


 public:
  SampleCtg(unsigned int _nRow, unsigned int _nCtg);
  ~SampleCtg();

  SplitPred *SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank) const;

  inline double SetNode(unsigned int sIdx, double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNux sNode;
    double ySum = sNode.Init(yVal, sCount, ctg);
    sampleNode[sIdx] = sNode;
    ctgRoot[ctg].Accum(ySum, sCount);

    return ySum;
  }
  
  
  void PreStage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const class RowRank *rowRank, std::vector<unsigned int> &row2Sample);
};


#endif
