// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file coprocrowrank.cc

   @brief Factory wrappers, parametrized by coprocessor state.

   @author Mark Seligman
 */

#include "coproc.h"
#include "rankedframe.h"


RankedFrame *RankedFrame::Factory(const Coproc *coproc,
				  const RLEFrame* rleFrame,
                                  double autoCompress,
				  PredictorT predPermute) {
  return new RankedFrame(rleFrame, autoCompress, predPermute);
}
