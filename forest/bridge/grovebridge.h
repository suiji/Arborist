// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grovebridge.h

   @brief Grove-training methods exportable to front end.

   @author Mark Seligman
 */


#ifndef FOREST_BRIDGE_GROVEBRIDGE_H
#define FOREST_BRIDGE_GROVEBRIDGE_H

#include <vector>
#include <memory>
#include <complex>

using namespace std;


struct GroveBridge {
  GroveBridge(unique_ptr<class Grove>);

  ~GroveBridge();
  
  
  /**
     @brief Getter for splitting information values.

     @return reference to per-predictor information vector.
   */
  const vector<double>& getPredInfo() const;
  
  /**
     @brief Main entry for training.
   */
  static unique_ptr<GroveBridge> train(const struct TrainBridge& trainBridge,
				       const struct SamplerBridge& sampler,
				       unsigned int treeOff,
				       unsigned int treeChunk,
				       const struct LeafBridge& leafBridge);


  const vector<size_t>& getNodeExtents() const;
  size_t getNodeCount() const;
  void dumpTree(complex<double> treeOut[]) const;
  void dumpScore(double scoreOut[]) const;


  const vector<size_t>& getFacExtents() const;


  /**
     @brief Passes through to Forest method.

     @return # bytes in current chunk of factors.
   */
  size_t getFactorBytes() const;


  /**
     @brief Dumps the splitting bits into a fixed-size raw buffer.
   */
  void dumpFactorRaw(unsigned char facOut[]) const;

  
  /**
     @brief Dumps the observed bits into a fixed-sized raw buffer.
   */
  void dumpFactorObserved(unsigned char obsOut[]) const;


private:

  const unique_ptr<class Grove> grove; ///< Core-level instantiation.
};

#endif
