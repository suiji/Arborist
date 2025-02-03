// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fepredict.h

   @brief Core handshake with prediction bridge.

   @author Mark Seligman
 */

#ifndef FOREST_FEPREDICT_H
#define FOREST_FEPREDICT_H

#include "typeparam.h"

#include <vector>

/**
   @brief Interface class for front end.

   Holds simulation-specific parameters of the data and constructs
   forest, leaf and diagnostic structures.
*/
struct FEPredict {

  /**
     @brief Initializes prediction state.
   */
  static void initPredict(bool indexing,
			  bool bagging,
			  unsigned int nPermute,
			  bool trapUnobserved);

  
  /**
     @brief Moves quantile vector to Quant.
   */
  static void initQuant(vector<double> quanitle);


  /**
     @brief Sets prediction reporting states.
   */
  static void initCtgProb(bool doProb);


  static void deInit();
};

#endif
