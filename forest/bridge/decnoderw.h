// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file decnoderw.h

   @brief Core-specific packing/unpacking of external TreeNode representations.

   @author Mark Seligman
 */

#ifndef FOREST_DECNODERW_H
#define FOREST_DECNODERW_H


#include "decnode.h"

#include <complex>
#include <memory>

struct DecNodeRW {

  /**
     @brief Unpacks nodes from a paired-double representation, such as complex.
   */
  static vector<vector<DecNode>> unpackNodes(const complex<double> nodes[],
					     const double nodeExtent[],
					     unsigned int nTree);

  
  /**
     @brief Builds a forest-wide score vector from R-internal format.
   */
  static vector<vector<double>> unpackScores(const double scores[],
					     const double nodeExtent[],
					     unsigned int nTree);


  static vector<unique_ptr<class BV>> unpackBits(const unsigned char raw[],
						 const double extent[],
						 unsigned int nTree);
};

#endif
