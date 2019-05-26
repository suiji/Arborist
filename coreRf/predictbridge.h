// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.h

   @brief Prediction methods exportable to front end.

   @author Mark Seligman
 */


#ifndef CORE_PREDICTBRIDGE_H
#define CORE_PREDICTBRIDGE_H

#include<vector>
#include<memory>
using namespace std;

struct PredictBridge {
  PredictBridge(unique_ptr<class Predict>);
  
  ~PredictBridge();

private:

  unique_ptr<class Predict> predict;
};


#endif
