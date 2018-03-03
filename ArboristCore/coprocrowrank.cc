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
#include "rowrank.h"


RowRank *RowRank::Factory(const Coproc *coproc,
			  const FrameTrain *frameTrain,
			  const unsigned int _feRow[],
			  const unsigned int _feRank[],
			  //		  const unsigned int *_numOffset,
			  //const double *_numVal,
			  const unsigned int _feRLE[],
			  unsigned int _feRLELength,
			  double _autoCompress) {
  return new RowRank(frameTrain, _feRow, _feRank, /*_numOffset, _numVal,*/ _feRLE, _feRLELength, _autoCompress);
}
