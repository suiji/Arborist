// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file bagRf.h

   @brief C++ interface to R entry for bagged rows.  There is no direct
   counterpart in Core, which records bagged rows using a bit matrix.

   @author Mark Seligman
 */

#ifndef RF_BAG_RF_H
#define RF_BAG_RF_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief Summary of bagged rows, by tree.
 */
class BagRf {
  const size_t nObs; // # rows trained.
  const unsigned int nTree; // # trees trained.
  const size_t rowBytes; // # count of raw bytes in summary object.
  RawVector raw; // Allocated OTF and moved.

 public:
  BagRf(size_t nObs_, unsigned int nTree_);
  BagRf(size_t nObs_, unsigned int nTree_, const RawVector &raw_);
  ~BagRf();

  /**
     @brief Getter for row count.
   */
  const auto getNObs() const {
    return nObs;
  }


  /**
     @brief Getter for tree count.
   */
  const auto getNTree() const {
    return nTree;
  }


  /**
     @brief Consumes a chunk of tree bags following training.

     @param train is the trained object.

     @param chunkOff is the offset of the current chunk.
   */
  void consume(const struct TrainChunk* train,
               unsigned int chunkOff);

  /**
     @brief Bundles trained bag into format suitable for front end.

     @return list containing raw data and summary information.
   */
  List wrap();

  /**
     @brief Reads bundled bag information in front-end format.

     @param sBag contains the training bag.

     @param sPredFrame summarizes the prediction data set.

     @param oob indicates whether a non-null bag is requested.

     @return instantiation containing bag raw data.
   */
  static unique_ptr<struct BagBridge> unwrap(const List& sBag,
                                            const List& sPredFrame,
                                            bool oob);

  
  /**
     @brief Checks that bag and prediction data set have conforming rows.

     @param sBag is the training bag.

     @param sPredFrame summarizes the prediction data set.
   */
  static SEXP checkOOB(const List& sBag,
                       const List& sPredFrame);
  

  /**
     @brief Reads bundled bag information for export.

     @param sBag contains the training bag.

     @return instantiation containing baga raw data.
   */
  static unique_ptr<struct BagBridge> unwrap(const List& sBag);
};

#endif
