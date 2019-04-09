// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file coproc.cc

   @brief Stubbed methods indicating lack of coprocessor support.

   @author Mark Seligman
 */

#include "coproc.h"


/**
   @brief
 */
unique_ptr<Coproc> Coproc::Factory(bool enable, vector<string> &diag) {
  diag.push_back("Executable built without coprocessor support.");

  return make_unique<Coproc>(0, 1);
}
