// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file trainbridge.cc

   @brief Exportable classes and methods from the Predict class.

   @author Mark Seligman
*/

#include "predictbridge.h"

#include "predict.h"
#include "forest.h"
#include "leaf.h"

PredictBridge::PredictBridge(unique_ptr<Predict> predict_) : predict(move(predict_)) {
}


PredictBridge::~PredictBridge() {
}
