// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CANDRF_H
#define SPLIT_CANDRF_H

/**
   @file cand.h

   @brief Manages non-specific splitting candidate selection.

   @author Mark Seligman

 */

#include "typeparam.h"

#include <vector>

class Cand {

public:
  
  virtual vector<class SplitNux>
  precandidates(class SplitFrontier* splitFrontier,
		const class Bottom* bottom) const;
};

#endif
