// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.h

   @brief Sampling functions.

   Reworks and enhances Nathan Russell's 2016 implementation for Rcpp.
   @author Mark Seligman
 */

#ifndef CORE_SAMPLE_H
#define CORE_SAMPLE_H

#include "prng.h"
#include "bheap.h"

#include <vector>
#include <numeric>

using namespace std;

namespace Sample {

  template<typename indexType>
  struct Walker {
    vector<double> weight;
    vector<indexType> coIndex;

    Walker(const double prob[],
	   indexType nObs) :
      weight(vector<double>(nObs)),
      coIndex(vector<indexType>(nObs)) {

      // Rescaling by 'nObs' enables conditional probability to be taken
      // when weighing slot.  Also appears to diminish rounding error.
      for (indexType i = 0; i < nObs; i++)
	weight[i] = prob[i] * nObs;

      vector<indexType> overMean, underMean;
      for (indexType i = 0; i < nObs; i++) {
	if (weight[i] < 1.0)
	  underMean.push_back(i);
	else
	  overMean.push_back(i);
      }

      for (indexType i = 0; i < nObs; i++) {
	// Rounding error may cause indices to be missed:
	// TODO:  randomize the scragglers.
	if (overMean.empty() || i == underMean.size())
	  break;
	indexType overIdx = overMean.back();
	indexType underIdx = underMean[i];
	coIndex[underIdx] = overIdx; // 'overIdx' may be reused.
	weight[overIdx] += (weight[underIdx] - 1.0);
	if (weight[overIdx] < 1.0) {
	  underMean.push_back(overIdx);
	  overMean.pop_back();
	}
      }
    }

    
    vector<size_t> sample(size_t nSamp) {
      vector<size_t> idxOut(nSamp);

      // Some implementions piggybacks index lookup with random weight
      // generation.  Separate random variates are drawn here to
      // improve resolution at high observation count.
      vector<size_t> rIndex = PRNG::rUnifIndex(nSamp, weight.size());
      vector<double> ru = PRNG::rUnif(nSamp);
      for (size_t i = 0; i < nSamp; i++) {
	size_t idx = rIndex[i];
	idxOut[i] = ru[i] < weight[idx] ? idx : coIndex[idx];
      }
      return idxOut;
    }
  };
  

  /**
   @brief Uniform sampling without replacement.

   @param sampleCoeff are the top nSamp-many scaling coefficients.

   Type currently fixed to size_t ut satisfy rUnifIndex().
   
   @param nObs is the sequence size from which to sample.

   @return vector of sampled indices.
 */
  template<typename indexType>
  vector<indexType> sampleUniform(const vector<size_t>& sampleScale,
				  indexType nObs) {
    vector<size_t> rn = PRNG::rUnifIndex(sampleScale);
    indexType nSamp = sampleScale.size();
    vector<indexType> idxSeq(nObs);
    vector<indexType> idxOut(nSamp);
    iota(idxSeq.begin(), idxSeq.end(), 0);
    for (indexType i = 0; i < nSamp; i++) {
      indexType index = rn[i];
      idxOut[i] = exchange(idxSeq[index], idxSeq[nObs - 1 - i]);
    }
    return idxOut;
  }


  /**
     @brief Permutes a zero-based set of contiguous values.

     @param nSlot is the number of values.

     @return vector of permuted indices.
   */
  template<typename indexType>
  vector<indexType> permute(indexType nSlot) {
    vector<double> vUnif = PRNG::rUnif(nSlot);
    BHeap<indexType> bHeap;
    for (auto variate : vUnif) {
      bHeap.insert(variate);
    }

    return bHeap.depopulate();
  }

  /**
     @brief Non-replacement sampling via Efraimidis-Spirakis.

     'nSamp' value cannot exceed 'nObs', and may be much smaller.
   */
  template<typename indexType>
  vector<indexType> sampleEfraimidis(const vector<double>& prob,
				     indexType nSamp = 0) {
    indexType nObs = prob.size();
    vector<double> vUnif = PRNG::rUnif(nObs);
    BHeap<indexType> bHeap;
    for (indexType slot = 0; slot < nObs; slot++) {
      bHeap.insert(-log(vUnif[slot]) / prob[slot]);
    }

    return bHeap.depopulate(nSamp == 0 ? nObs : nSamp);
  }
};

#endif
