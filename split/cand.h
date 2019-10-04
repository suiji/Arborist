// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CAND_H
#define SPLIT_CAND_H

/**
   @file cand.h

   @brief Manages non-specific splitting candidate selection.

   @author Mark Seligman

 */

#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

/**
   @brief Minimal information needed to preschedule a splitting candidate.
 */

class Cand {
public:
  
  virtual vector<DefCoord>
  precandidates(class SplitFrontier* splitFrontier,
		const class DefMap* bottom) const;
};

#endif
