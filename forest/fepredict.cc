// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fepredict.cc

   @brief Bridge entry to static initializations.

   @author Mark Seligman
*/

#include "treenode.h"
#include "fecore.h"
#include "fepredict.h"
#include "predict.h"
#include "quant.h"


void FEPredict::initPredict(bool indexing,
			    bool bagging,
			    unsigned int nPermute,
			    bool trapUnobserved) {
  ForestPrediction::init(indexing);
  Predict::init(bagging, trapUnobserved, nPermute);
}


void FEPredict::initQuant(vector<double> quantile) {
  Quant::init(std::move(quantile));
}


void FEPredict::initCtgProb(bool doProb) {
  CtgProb::init(doProb);
}


void FEPredict::deInit() {
  Predict::deInit();
  ForestPrediction::deInit();
  Quant::deInit();
  CtgProb::deInit();
  FECore::deInit();
}
