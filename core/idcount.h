// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file idcount.h

   @brief Class definition for identifier/count container.

   @author Mark Seligman
 */

#ifndef CORE_IDCOUNT_H
#define CORE_IDCOUNT_H

#include "typeparam.h"


#include <vector>
using namespace std;

/**
   @brief Identifier, count for decompressed samples.
 */
struct IdCount {
  IndexT id; ///< e.g., index or rank.
  IndexT sCount; ///< count.

  
  IdCount(IndexT id_,
          unsigned int sCount_) : id(id_), sCount(sCount_) {
  }


  IdCount() : id(0), sCount(0) {}


  auto getId() const {
    return id;
  }


  auto getSCount() const {
    return sCount;
  }
};


#endif
