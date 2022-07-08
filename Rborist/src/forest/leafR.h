// Copyright (C)  2012-2022   Mark Seligman
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
   @file leafR.h

   @brief C++ interface to R entry for sampled leaves.

   @author Mark Seligman
 */

#ifndef FOREST_LEAF_R_H
#define FOREST_LEAF_R_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief Summary of leaf samples.
 */
struct LeafR {
  static const string strExtent;
  static const string strIndex;

  size_t extentTop; // " " leaf extent buffer.
  size_t indexTop;  // " " sample index buffer.

  NumericVector extent; // Leaf extents.
  NumericVector index; // Sample indices.

  LeafR();

  /**
     @brief Bundles trained leaf into format suitable for R.

     Wrap functions are called from TrainR::summary, following which 'this' is
     deleted.  There is therefore no need to initialize the extent and index
     state.
     
   */
  List wrap();

  
  /**
     @brief Consumes a block of samples following training.

     @param scale is a fudge-factor for resizing.
   */
  void bridgeConsume(const struct LeafBridge* sb,
		     double scale);

  static unique_ptr<struct LeafBridge> unwrap(const List& lLeaf,
					      const struct SamplerBridge* samplerBridge);
};

#endif
