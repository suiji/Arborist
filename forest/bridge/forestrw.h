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

#include <vector>
using namespace std;


struct ForestRW {
  static void dump(const class Forest* forest,
		   vector<vector<unsigned int> >& predTree,
		   vector<vector<double> >& splitTree,
		   vector<vector<size_t> >& lhDelTree,
		   vector<vector<unsigned char> >& facSplitTree,
		   vector<vector<double>>& scoreTree);
};

#endif
