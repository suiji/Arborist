// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestrw.h

   @brief Core-specific packing/unpacking of external Forest representations.

   @author Mark Seligman
 */

#ifndef FOREST_BRIDGE_FORESTRRW_H
#define FOREST_BRIDGE_FORESTRRW_H

#include "typeparam.h"
#include "dectree.h"

#include <complex>
#include <vector>

using namespace std;


struct ForestRW {

  static vector<DecTree> unpackDecTree(unsigned int nTree,
				const double nodeExtent[],
				const complex<double> nodes[],
				const double score[],
				const double facExtent[],
				const unsigned char facSplit[],
				const unsigned char facObserved[]);


  static vector<double> unpackDoubles(const double val[],
				      const size_t extent);


  static BV unpackBits(const unsigned char raw[],
		       size_t extent);

  
  static vector<DecNode> unpackNodes(const complex<double> nodes[],
				     size_t extent);

  static class Leaf unpackLeaf(const struct SamplerBridge& samplerBridge,
			       const double extent_[],
			       const double index_[]);


  static vector<vector<size_t>> unpackExtent(const struct SamplerBridge& samplerBridge,
				      const double extentNum[]);


  static vector<vector<vector<size_t>>> unpackIndex(const struct SamplerBridge& samplerBridge,
					     const vector<vector<size_t>>& extent,
					     const double numVal[]);


  static void dump(const class Forest* forest,
		   vector<vector<unsigned int> >& predTree,
		   vector<vector<double> >& splitTree,
		   vector<vector<size_t> >& lhDelTree,
		   vector<vector<unsigned char> >& facSplitTree,
		   vector<vector<double>>& scoreTree);
};

#endif
