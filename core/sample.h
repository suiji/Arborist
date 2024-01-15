// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.h

   @brief Sampling functions.

   Reworks and extends Nathan Russell's 2016 implementation for Rcpp.
   
   @author Mark Seligman
 */

#ifndef CORE_SAMPLE_H
#define CORE_SAMPLE_H

#include "prng.h"
#include "bheap.h"

#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

namespace Sample {

  template<typename indexType>
  struct Walker {
    vector<double> weight;
    vector<indexType> coIndex;

    Walker(const vector<double> prob,
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


    vector<indexType> sample(indexType nSamp,
			  const vector<indexType>& obsOmit) {
      vector<indexType> idxOut(nSamp);

      // Some implementions piggyback index lookup with random weight
      // generation.  Separate random variates are drawn here to
      // improve resolution at high observation count.
      vector<indexType> rIndex = PRNG::rUnifIndex<indexType>(nSamp, weight.size());
      vector<double> ru = PRNG::rUnif(nSamp);
      for (indexType i = 0; i < nSamp; i++) {
	indexType idx = rIndex[i];
	idxOut[i] = ru[i] < weight[idx] ? idx : coIndex[idx];
      }
      return idxOut;
    }
  };


  template<typename indexType>
  vector<indexType> sampleWith(indexType nObs,
			       const vector<indexType>& omitMap,
			       indexType nSamp) {
    if (omitMap.empty())
      return PRNG::rUnifIndex<indexType>(nSamp, nObs);
    else
      return PRNG::rIndexScatter(nSamp, omitMap);
  }
  

  /**
     @brief Orders indices with omitted values placed last.

     @param nObs is the # indices to order.

     @param omit are indices to be omitted.
     
     @return sequential indices with omitted placed at the end.
   */
  template<typename indexType>
  vector<indexType> omitIndices(indexType nObs,
				const vector<indexType>& omit) {
  BHeap<indexType> bHeap;
  for (const size_t& idx : omit) {
    bHeap.insert(idx);
  }
  vector<indexType> idxEligible(nObs);
  iota(idxEligible.begin(), idxEligible.end(), 0);
  indexType idxEnd = nObs;
  while (!bHeap.empty()) { // Omits indices in descending order.
    idxEnd--;
    indexType idx = bHeap.pop();
    idxEligible[idxEnd] = idx;
    idxEligible[idx] = idxEnd;
  }
  return idxEligible;
}


  template<typename indexType>
  vector<indexType> scaleVariates(indexType idxEnd,
				 indexType nSamp) {
    vector<indexType> sampleScale(nSamp);
    iota(sampleScale.begin(), sampleScale.end(), idxEnd - nSamp + 1);
    reverse(sampleScale.begin(), sampleScale.end());
    return PRNG::rUnifIndex<indexType>(sampleScale);
  }


  /**
   @brief Uniform sampling without replacement.

   @param nObs is the number of indices to sample.

   @param nSamp is the # samples to draw.

   @param omit is a set of indices held out from sampling.

   @return vector of sampled indices.
 */
  template<typename indexType>
  vector<indexType> sampleWithout(indexType nObs,
				  const vector<indexType>& omit,
				  indexType nSamp) {
    vector<indexType> indices = omitIndices(nObs, omit);
    size_t idxEnd = nObs - omit.size() - 1;
    vector<indexType> rn = scaleVariates(idxEnd, nSamp);
    vector<indexType> idxOut(nSamp);
    for (indexType i = 0; i < nSamp; i++) {
      indexType index = rn[i];
      idxOut[i] = exchange(indices[index], indices[idxEnd - i]);
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

     @param nSamp value <= prob.size(), potentially <<.
   */
  template<typename indexType>
  vector<indexType> sampleEfraimidis(const vector<double>& prob,
				     const vector<indexType>& obsOmit,
				     indexType nSamp = 0) {
    vector<double> vUnif = PRNG::rUnif(prob.size());
    const double* variate = &vUnif[0];
    BHeap<indexType> bHeap;
    for (const double& probability : prob) {
      if (probability > 0) {
	bHeap.insert(-log(*variate / probability));
      }
      variate++;
    }

    return bHeap.depopulate(nSamp);
  }
}

#endif
