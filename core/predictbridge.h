// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.h

   @brief Data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef CORE_PREDICT_BRIDGE_H
#define CORE_PREDICT_BRIDGE_H

#include "block.h"

#include <vector>
#include <memory>

using namespace std;

/**
   @brief Consolidates common components required by all prediction entries.

   These are typically unwrapped by the front end from several data structures.
 */
struct PredictBridge {
  /**
     @brief Constructor boxes training and output summaries.

     @param nThread is the number of OMP threads requested.

     Remaining parameters mirror similarly-named members.
   */
  PredictBridge(bool oob,
                unique_ptr<class ForestBridge> forest_,
                unique_ptr<class BagBridge> bag_,
                unique_ptr<class LeafBridge> leaf_,
                unsigned int nThread);

  PredictBridge(bool oob,
                unique_ptr<class ForestBridge> forest_,
                unique_ptr<class BagBridge> bag_,
                unique_ptr<class LeafBridge> leaf_,
                const vector<double>& quantile,
                unsigned int nThread);

  ~PredictBridge();

  /**
     @brief Gets an acceptable block row count.

     @param rowCount is a requested count.

     @return count of rows in block.
   */
  static size_t getBlockRows(size_t rowCount);


  /**
     @brief Predicts over a block of observations.

     @param blockNum collects numerical observations.

     @param blockFac collects factor-valued observations.

     @param row is the beginning row index of the block.
   */
  void predictBlock(const BlockDense<double>* blockNum,
                    const BlockDense<unsigned int>* blockFac,
                    size_t row) const;

  const class Quant* getQuant() const;
  
  LeafBridge* getLeaf() const;

private:
  unique_ptr<class BagBridge> bag;
  unique_ptr<class ForestBridge> forest;
  unique_ptr<class LeafBridge> leaf;
  unique_ptr<class Quant> quant;
  unique_ptr<class Predict> predictCore;
};


#endif
