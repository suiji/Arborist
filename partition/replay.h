// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file replay.h

   @brief Encodes L/R partitioning of frontier.

   @author Mark Seligman

 */

#ifndef PARTITION_REPLAY_H
#define PARTITION_REPLAY_H

class Replay {
  unique_ptr<class BV> explicit;  // Whether index be explicitly replayed.
  unique_ptr<class BV> left;  // Explicit:  L/R ; else undefined.
};

#endif
