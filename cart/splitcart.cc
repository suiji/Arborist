// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitcart.cc

   @brief Directs splitting via accumulators.

   @author Mark Seligman
 */


#include "splitfrontier.h"
#include "sfcart.h"
#include "splitcart.h"
#include "frontier.h"
#include "booster.h" // TEMPORARY

unique_ptr<SplitFrontier> SplitCart::factory(Frontier* frontier) {
  // Splitter should be dispatched upstream of here to avoid
  // having to test for boosting:
  if (frontier->getNCtg() > 0 && !Booster::boosting()) {
    return make_unique<SFCtgCart>(frontier);
  }
  else {
    return make_unique<SFRegCart>(frontier);
  }
}
