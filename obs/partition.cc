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

#include "partition.h"
#include "predictorframe.h"
#include "splitnux.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
ObsPart::ObsPart(const PredictorFrame* layout,
		 IndexT bagCount_) :
  bagCount(bagCount_),
  bufferSize(layout->getSafeSize(bagCount)),
  stageRange(layout->getNPred()) {
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


IndexT* ObsPart::getIdxBuffer(const SplitNux& nux) const {
  return idxBuffer(nux.getStagedCell());
}


Obs* ObsPart::getBuffers(const SplitNux& nux, IndexT*& sIdx) const {
  return buffers(nux.getStagedCell(), sIdx);
}


Obs* ObsPart::getPredBase(const SplitNux& nux) const {
  return getPredBase(nux.getStagedCell());
}


IndexT ObsPart::getSampleIndex(const SplitNux& cand,
			       IndexT obsIdx) const {
  return idxBuffer(cand.getStagedCell())[obsIdx];
}


void ObsPart::restageDiscrete(const PathT* prePath,
			      const StagedCell& mrra,
			      vector<IndexT>& obsScatter) {
  Obs *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  for (IndexT idx = mrra.obsRange.getStart(); idx < mrra.obsRange.getEnd(); idx++) {
    PathT path = prePath[idx];
    if (NodePath::isActive(path)) {
      IndexT obsDest = obsScatter[path]++;
      srTarg[obsDest] = srSource[idx];
      idxTarg[obsDest] = idxSource[idx];
    }
  }
}


void ObsPart::restageTied(const PathT* prePath,
			  vector<IndexT>& runCount,
			  const StagedCell& mrra,
			  vector<IndexT>& obsScatter) {
  Obs *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  IndexT rankIdx = 0;
  vector<IndexT> idxPrev(runCount.size());
  fill(idxPrev.begin(), idxPrev.end(), mrra.getRunCount());
  srSource[mrra.obsRange.getStart()].setTie(true); // Fillip;  temporary.
  for (IndexT idx = mrra.obsRange.getStart(); idx != mrra.obsRange.getEnd(); idx++) {
    Obs sourceNode = srSource[idx];
    rankIdx += sourceNode.isTied() ? 0 : 1;
    PathT path = prePath[idx];
    if (NodePath::isActive(path)) {
      if (rankIdx != idxPrev[path]) {
	sourceNode.setTie(false);
	runCount[path]++;
	idxPrev[path] = rankIdx;
      }
      else {
	sourceNode.setTie(true);
      }
      IndexT obsDest = obsScatter[path]++;
      srTarg[obsDest] = sourceNode;
      idxTarg[obsDest] = idxSource[idx];
    }
  }
}


void ObsPart::restageValues(const PathT* prePath,
			    vector<IndexT>& runCount,
			    const StagedCell& mrra,
			    vector<IndexT>& obsScatter,
			    vector<IndexT>& valScatter,
			    const vector<IndexT>& valSource,
			    vector<IndexT>& valTarg) {
  Obs *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  vector<IndexT> idxPrev(runCount.size());
  fill(idxPrev.begin(), idxPrev.end(), mrra.valIdx + mrra.getRunCount());
  IndexT rankIdx = mrra.valIdx;
  srSource[mrra.obsRange.getStart()].setTie(true); // Fillip;  temporary.
  for (IndexT idx = mrra.obsRange.getStart(); idx != mrra.obsRange.getEnd(); idx++) {
    Obs sourceNode = srSource[idx];
    rankIdx += sourceNode.isTied() ? 0 : 1;
    PathT path = prePath[idx];
    if (NodePath::isActive(path)) {
      if (rankIdx != idxPrev[path]) {
	sourceNode.setTie(false);
	runCount[path]++;
	idxPrev[path] = rankIdx;
	IndexT valDest = valScatter[path]++;
	valTarg[valDest] = valSource[rankIdx];
      }
      else {
	sourceNode.setTie(true);
      }
      IndexT obsDest = obsScatter[path]++;
      srTarg[obsDest] = sourceNode;
      idxTarg[obsDest] = idxSource[idx];
    }
  }
}
