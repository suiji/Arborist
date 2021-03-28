// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sampler.h

   @brief Forest-wide packed representation of sampled observations.

   @author Mark Seligman
 */

#ifndef FOREST_SAMPLERCRESC_H
#define FOREST_SAMPLERCRESC_H

#include "typeparam.h"
#include "samplernux.h"
#include "sampler.h"
#include "sample.h"
#include "leaf.h"

#include <vector>
#include <memory>

/**
   @brief SamplerNux block for crescent frame.
 */
class SamplerCresc {
  vector<SamplerNux> samplerNux;
  const vector<double> yProxy; // Only employed for categorical response.
  unique_ptr<class Leaf> leaf; // Subclassed leaf type.
  unique_ptr<class Sample> sample; // Reset at each tree.
  vector<size_t> height;

public:
  SamplerCresc(const vector<double>& yNum,
	       unsigned int treeChunk);


  SamplerCresc(const vector<PredictorT>& yCtg,
	       PredictorT nCtg,
	       const vector<double>& yProxy,
	       unsigned int treeChunk);


  class Sample* getSample() const;

  
  void rootSample(const class TrainFrame* frame);

  
  vector<double> bagLeaves(const vector<IndexT>& leafMap,
			   unsigned int tIdx);


  const vector<size_t>& getHeight() const {
    return height;
  }

  
  /**
     @brief Records multiplicity and leaf index for bagged samples
     within a tree.  Accessed by bag vector, so sample indices must
     reference consecutive bagged rows.
     @param leafMap maps sample indices to leaves.
  */
  void bagLeaves(const class Sample *sample,
                 const vector<IndexT> &leafMap,
		 unsigned int tIdx);


  void dumpRaw(unsigned char blRaw[]) const; 
};

#endif
