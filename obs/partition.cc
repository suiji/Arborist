// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file partition.cc

   @brief Methods to repartition observation frame by tree node.

   @author Mark Seligman
 */

#include "obsfrontier.h"
#include "partition.h"
#include "layout.h"
#include "sampleobs.h"
#include "splitnux.h"
#include "path.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
ObsPart::ObsPart(const Layout* layout,
		 IndexT bagCount_) :
  bagCount(bagCount_),
  bufferSize(layout->getSafeSize(bagCount)),
  pathIdx(bufferSize),
  stageRange(layout->getNPred()),
  noRank(layout->getNoRank()) {
  indexBase = new IndexT[2* bufferSize];
  obsCell = new Obs[2 * bufferSize];

  // Coprocessor variants:
  //  vector<unsigned int> destRestage(bufferSize);
  //  vector<unsigned int> destSplit(bufferSize);
}


/**
  @brief Base class destructor.
 */
ObsPart::~ObsPart() {
  delete [] obsCell;
  delete [] indexBase;
}


IndexT* ObsPart::getIdxBuffer(const SplitNux* nux) const {
  return idxBuffer(nux->getStagedCell());
}


Obs* ObsPart::getBuffers(const SplitNux& nux, IndexT*& sIdx) const {
  return buffers(nux.getStagedCell(), sIdx);
}


Obs* ObsPart::getPredBase(const SplitNux* nux) const {
  return getPredBase(nux->getStagedCell());
}
