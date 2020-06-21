// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CAND_H
#define SPLIT_CAND_H

/**
   @file candcart.h

   @brief Manages generic splitting candidate selection.

   @author Mark Seligman

 */

#include "cand.h"
#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

/**
   @brief Minimal information needed to preschedule a splitting candidate.
 */

struct Cand {
  static vector<PreCand> precandidates(class SplitFrontier* splitFrontier,
				       class DefMap* bottom);
};

#endif
