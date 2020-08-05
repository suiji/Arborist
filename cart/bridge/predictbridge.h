// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.h

   @brief Bridge data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef CART_BRIDGE_PREDICTBRIDGE_H
#define CART_BRIDGE_PREDICTBRIDGE_H

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
  PredictBridge(unique_ptr<struct RLEFrame> rleFrame_,
                unique_ptr<struct ForestBridge> forest_,
                unique_ptr<struct BagBridge> bag_,
                unique_ptr<struct LeafBridge> leaf_,
		bool importance,
                unsigned int nThread);

  PredictBridge(unique_ptr<struct RLEFrame> rleFrame_,
                unique_ptr<struct ForestBridge> forest_,
                unique_ptr<struct BagBridge> bag_,
                unique_ptr<struct LeafBridge> leaf_,
		bool importance,
                const vector<double>& quantile,
                unsigned int nThread);

  ~PredictBridge();


  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  void predict() const;


  struct LeafBridge* getLeaf() const;

  
  /**
     @return vector of predection quantiles iff quant non-null else empty.
   */
  const vector<double> getQPred() const;

  /**
     @return vector of estimate quantiles iff quant non-null else empty.
   */
  const vector<double> getQEst() const;
  
private:
  unique_ptr<struct RLEFrame> rleFrame;
  unique_ptr<struct BagBridge> bagBridge;
  unique_ptr<struct ForestBridge> forestBridge;
  unique_ptr<struct LeafBridge> leafBridge;
  const bool importance; // Whether permutation importance is requested.
  unique_ptr<class Quant> quant;
};

#endif
