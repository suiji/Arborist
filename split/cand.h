// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CAND_H
#define SPLIT_CAND_H

/**
   @file cand.h

   @brief Manages generic splitting candidate selection.

   @author Mark Seligman
 */


#include "typeparam.h"

#include <vector>

/**
   @brief Minimal information needed to preschedule a splitting candidate.
 */

struct Cand {
  static void precandidates(class DefFrontier* defFrontier);
};

#endif
