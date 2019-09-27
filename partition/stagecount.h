// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file stagecount.h

   @brief Staging summary.

   @author Mark Seligman
 */

#ifndef PARTITION_STAGECOUNT_H
#define PARTITION_STAGECOUNT_H


#include "typeparam.h"

/**
   @brief Summarizes staging operation.
 */
struct StageCount {
  IndexT expl;
  bool singleton;
};

#endif
