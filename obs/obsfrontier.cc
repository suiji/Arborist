// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obsfrontier.cc

   @brief Methods involving individual definition layers.

   @author Mark Seligman
 */

#include "splitnux.h"
#include "predictorframe.h"
#include "frontier.h"
#include "obsfrontier.h"
#include "path.h"
#include "interlevel.h"
#include "partition.h"
#include "samplemap.h"
#include "sampledobs.h"
#include "indexset.h"
#include "algparam.h"


ObsFrontier::ObsFrontier(const Frontier* frontier_,
			 InterLevel* interLevel_) :
  frontier(frontier_),
  interLevel(interLevel_),
  nPred(interLevel->getNPred()),
  nSplit(interLevel->getNSplit()),
  node2Front(vector<IndexRange>(nSplit)), // Initialized to empty.
  stagedCell(vector<vector<StagedCell>>(nSplit)),
  stageCount(0),
  runCount(0),
  layerIdx(0), // Not on layer yet, however.
  nodePath(backScale(nSplit)) {
  NodePath::setNoSplit(frontier->getBagCount());
  // Coprocessor only.
  // LiveBits df;
  //  fill(mrra.begin(), mrra.end(), df);
}


void ObsFrontier::prestageRoot(const PredictorFrame* frame,
			       const SampledObs* sampledObs) {
  for (PredictorT predIdx = 0; predIdx != nPred; predIdx++) {
    interLevel->setStaged(0, predIdx, predIdx);
    stagedCell[0].emplace_back(predIdx, runCount, frontier->getBagCount(), sampledObs->getRunCount(predIdx));
    runCount += stagedCell[0].back().trackRuns ? sampledObs->getRunCount(predIdx) : 0;
  }
  stageCount = nPred;
  runValues();
}


void ObsFrontier::runValues() {
  runValue = vector<IndexT>(runCount);
}


void ObsFrontier::prestageAncestor(ObsFrontier* ofFront,
				   IndexT nodeIdx,
				   PredictorT stagePosition) {
  IndexT ancIdx = front2Node[nodeIdx]; // Predecessor index at this level.
  ofFront->prestageRange(stagedCell[ancIdx][stagePosition], node2Front[ancIdx]);
  interLevel->appendAncestor(stagedCell[ancIdx][stagePosition], layerIdx);
}


void ObsFrontier::prestageRange(const StagedCell& cell,
				const IndexRange& range) {
  for (IndexT nodeIdx = range.getStart(); nodeIdx != range.getEnd(); nodeIdx++) {
    interLevel->setStaged(nodeIdx, cell.getPredIdx(), stagedCell[nodeIdx].size());
    stagedCell[nodeIdx].emplace_back(nodeIdx, cell, runCount, frontier->getNodeRange(nodeIdx));
    runCount += cell.trackRuns ? min(cell.runCount, cell.obsRange.idxExtent) : 0;
  }
  stageCount += range.getExtent();
}


void ObsFrontier::prestageLayer(ObsFrontier* ofFront) {
  IndexT nodeIdx = 0;
  for (vector<StagedCell>& nodeCells : stagedCell) {
    for (StagedCell& cell : nodeCells) {
      if (cell.isLive()) { // Otherwise already delisted.
	ofFront->prestageRange(cell, node2Front[nodeIdx]);
	interLevel->appendAncestor(cell, layerIdx);
      }
    }
    nodeIdx++;
  }
}


IndexT ObsFrontier::countLive() const {
  IndexT liveCount = 0;
  for (vector<StagedCell> nodeCells : stagedCell) {
    for (StagedCell& cell : nodeCells) {
      if (cell.isLive()) { // Otherwise already delisted.
	liveCount++;
      }
    }
  }
  return liveCount;
}


void ObsFrontier::setFrontRange(const vector<IndexSet>& frontierNodes,
				const vector<IndexSet>& frontierNext,
				IndexT endIdx) {
  front2Node = vector<IndexT>(frontierNext.size());
  IndexT terminalCount = 0;
  for (IndexT parIdx = 0; parIdx < frontierNodes.size(); parIdx++) {
    if (frontierNodes[parIdx].isTerminal()) {
      terminalCount++;
      delistNode(parIdx);
    }
    else {
      setFrontRange(frontierNext, parIdx, IndexRange(2 * (parIdx - terminalCount), 2), endIdx);
    }
  }
}


void ObsFrontier::setFrontRange(const vector<IndexSet>& frontierNext,
				IndexT nodeIdx,
				const IndexRange& range,
				IndexT endIdx) {
  node2Front[nodeIdx] = range;
  NodePath* pathBase = &nodePath[backScale(nodeIdx)];
  for (IndexT frontIdx = range.getStart(); frontIdx != range.getEnd(); frontIdx++) {
    pathInit(pathBase, frontierNext[frontIdx], endIdx);
    front2Node[frontIdx] = nodeIdx;
  }
}


void ObsFrontier::pathInit(NodePath* pathBase, const IndexSet& iSet, IndexT endIdx) {
  pathBase[iSet.getPath(pathMask())].init(iSet, endIdx);
}


void ObsFrontier::applyFront(const ObsFrontier* ofFront,
			     const vector<IndexSet>& frontierNext,
			     IndexT endIdx) {
  layerIdx++;
  nodePath = vector<NodePath>(backScale(nSplit));
  front2Node = vector<IndexT>(frontierNext.size());

  IndexT succStart = 0; // Loop-carried.
  for (IndexT nodeIdx = 0; nodeIdx < nSplit; nodeIdx++) {
    IndexRange range = node2Front[nodeIdx];
    if (range.empty())
      continue;
    IndexT succCount = 0;
    for (IndexT succFront = range.getStart(); succFront != range.getEnd(); succFront++) {
      succCount += ofFront->getFrontRange(succFront).getExtent();
    }
    IndexRange frontRange = IndexRange(succStart, succCount);
    succStart += succCount;
    if (frontRange.empty()) { // Newly extinct path:  flush rank arrays.
      delistNode(nodeIdx);
    }
    else {
      setFrontRange(frontierNext, nodeIdx, frontRange, endIdx);
    }
    node2Front[nodeIdx] = frontRange;
  }
}


void ObsFrontier::delistNode(IndexT nodeIdx) {
  for (StagedCell& cell : stagedCell[nodeIdx]) {
    if (cell.isLive()) {
      cell.delist();
      stageCount--;
    }
  }
}


unsigned int ObsFrontier::stage(PredictorT predIdx,
				ObsPart* obsPart,
				const PredictorFrame* frame,
				const SampledObs* sampledObs) {
  obsPart->setStageRange(predIdx, frame->getSafeRange(predIdx, frontier->getBagCount()));
  StagedCell& cell = stagedCell[0][predIdx];
  const IndexT rankImplicit = frame->getImplicitRank(predIdx);
  const IndexT rankMissing = frame->getMissingRank(predIdx);
  IndexT obsMissing = 0;
  IndexT* sIdx;
  Obs* srStart = obsPart->buffers(predIdx, 0, sIdx);
  Obs* spn = srStart;
  IndexT rankPrev = interLevel->getNoRank();
  IndexT valIdx = cell.valIdx;
  for (auto rle : frame->getRLE(predIdx)) {
    IndexT rank = rle.val;
    if (rank != rankImplicit) {
      for (IndexT row = rle.row; row != rle.row + rle.extent; row++) {
	IndexT smpIdx;
	SampleNux sampleNux;
	if (sampledObs->isSampled(row, smpIdx, sampleNux)) {
	  bool tie = rank == rankPrev;
	  spn++->join(sampleNux, tie);
	  *sIdx++ = smpIdx;
	  if (!tie) {
	    rankPrev = rank;
	    runCount++;
	    if (cell.trackRuns)
	      runValue[valIdx++] = rank;
	  }
	  if (rank == rankMissing)
	    obsMissing++;
	}
      }
    }
    else {
      cell.preResidual = spn - srStart;
    }
  }
  //  cout << "Predictor " << predIdx << ":  " << obsMissing << " missing " << ", " << spn - srStart << " observed" << endl;
  cell.updateCounts(frontier->getBagCount() - (spn - srStart), obsMissing);

  if (!cell.splitable()) {
    interLevel->delist(cell.coord);
    cell.delist();
    return 1;
  }
  else
    return 0;
}


unsigned int ObsFrontier::restage(ObsPart* obsPart,
				  const StagedCell& mrra,
				  ObsFrontier* ofFront) const {
  vector<StagedCell*> tcp(backScale(1));
  fill(tcp.begin(), tcp.end(), nullptr);

  vector<IndexT> runCount(backScale(1));
  const PathT* prePath = interLevel->getPathBlock(mrra.getPredIdx());

  // Run tracking is currently disabled, as no performance advantage
  // has been observed.  The main benefit to tracking run values is
  // the reduction in irregular accesses when setting run-based
  // splitting criteria, which consists of setting a large number of
  // bits indexed by irregular samples.  Run tracking enables thee
  // bit indices to be looked up directly from the run accumulator.
  if (mrra.trackRuns) {
    vector<IndexT> valScatter(backScale(1));
    vector<IndexT> obsScatter = packTargets(obsPart, mrra, tcp, valScatter);
    obsPart->restageValues(prePath, runCount, mrra, obsScatter, valScatter, runValue, ofFront->runValue);
  }
  else {
    vector<IndexT> obsScatter = packTargets(obsPart, mrra, tcp);
    if (mrra.trackableTies()) {
      obsPart->restageTied(prePath, runCount, mrra, obsScatter);
    }
    else {
      obsPart->restageDiscrete(prePath, mrra, obsScatter);
    }
  }

  unsigned int nExtinct = 0;

  // Speculatively assumes mrra has residual:
  for (PathT path = 0; path != backScale(1); path++) {
    StagedCell* cell = tcp[path];
    if (cell != nullptr) {
      cell->setRunCount(runCount[path]);
      if (!cell->splitable()) {
	interLevel->delist(cell->coord);
	cell->delist();
	nExtinct++;
      }
    }
  }

  return nExtinct;
}


// Successors may or may not themselves be dense.
vector<IndexT> ObsFrontier::packTargets(ObsPart* obsPart,
					const StagedCell& mrra,
					vector<StagedCell*>& tcp) const {
  vector<IndexT> preResidual(backScale(1));
  vector<IndexT> obsMissing(backScale(1));
  vector<IndexT> pathCount = pathRestage(obsPart, preResidual, obsMissing, mrra);
  vector<IndexT> obsScatter(backScale(1));
  IndexT idxStart = mrra.obsRange.getStart();
  const NodePath* pathPos = &nodePath[backScale(mrra.getNodeIdx())];
  PredictorT predIdx = mrra.getPredIdx();
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexT frontIdx;
    if (pathPos[path].getFrontIdx(frontIdx)) {
      IndexT extentDense = pathCount[path];
      tcp[path] = interLevel->getFrontCellAddr(SplitCoord(frontIdx, predIdx));
      tcp[path]->updatePath(idxStart, extentDense, preResidual[path], obsMissing[path]);
      obsScatter[path] = idxStart;
      idxStart += extentDense;
    }
  }
  return obsScatter;
}


vector<IndexT> ObsFrontier::packTargets(ObsPart* obsPart,
					const StagedCell& mrra,
					vector<StagedCell*>& tcp,
					vector<IndexT>& valScatter) const {
  vector<IndexT> preResidual(backScale(1));
  vector<IndexT> obsMissing(backScale(1));
  vector<IndexT> pathCount = pathRestage(obsPart, preResidual, obsMissing, mrra);
  vector<IndexT> obsScatter(backScale(1));
  IndexT idxStart = mrra.obsRange.getStart();
  const NodePath* pathPos = &nodePath[backScale(mrra.getNodeIdx())];
  PredictorT predIdx = mrra.getPredIdx();
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexT frontIdx;
    if (pathPos[path].getFrontIdx(frontIdx)) {
      IndexT extentDense = pathCount[path];
      tcp[path] = interLevel->getFrontCellAddr(SplitCoord(frontIdx, predIdx));
      tcp[path]->updatePath(idxStart, extentDense, preResidual[path], obsMissing[path]);
      obsScatter[path] = idxStart;
      valScatter[path] = tcp[path]->valIdx;
      idxStart += extentDense;
    }
  }
  return obsScatter;
}


vector<IndexT> ObsFrontier::pathRestage(ObsPart* obsPart,
					vector<IndexT>& preResidual,
					vector<IndexT>& obsMissing,
					const StagedCell& mrra) const {
  IndexRange obsRange = mrra.obsRange;
  const IdxPath* idxPath = interLevel->getRootPath();
  const IndexT* indexVec = obsPart->idxBuffer(&mrra);
  PathT* prePath = interLevel->getPathBlock(mrra.getPredIdx());
  vector<IndexT> pathCount(backScale(1));

  // Loop can be streamlined if mrra has no implicit observations
  // and no missing data.
  bool cutSeen = mrra.implicitObs() ? false : true;
  bool naSeen = false;
  IndexT threshResidual = obsRange.idxStart + mrra.preResidual;
  IndexT threshMissing = obsRange.getEnd() - mrra.obsMissing;
  for (IndexT idx = obsRange.getStart(); idx != obsRange.getEnd(); idx++) {
    cutSeen = cutSeen || (idx >= threshResidual);
    naSeen = naSeen || (idx >= threshMissing);
    PathT path;
    if (idxPath->pathSucc(indexVec[idx], pathMask(), path)) {
      pathCount[path]++;
      if (!cutSeen)
	preResidual[path]++;
      if (naSeen)
	obsMissing[path]++;
    }
    prePath[idx] = path;
  }

  return pathCount;
}


void ObsFrontier::updateMap(const IndexSet& iSet,
			    const BranchSense& branchSense,
			    const SampleMap& smNonterm,
			    SampleMap& smTerminal,
			    SampleMap& smNext) {
  if (!iSet.isTerminal()) {
    updateLive(branchSense, iSet, smNonterm, smNext);
  }
  else {
    updateExtinct(iSet, smNonterm, smTerminal);
  }
}


void ObsFrontier::updateLive(const BranchSense& branchSense,
			     const IndexSet& iSet,
			     const SampleMap& smNonterm,
			     SampleMap& smNext) {
  IndexT nodeIdx = iSet.getIdxNext();
  IndexT destTrue = smNext.range[nodeIdx].getStart();
  IndexT destFalse = smNext.range[nodeIdx+1].getStart();
  IndexRange range = smNonterm.range[iSet.getSplitIdx()];
  bool implicitTrue = !iSet.encodesTrue();
  for (IndexT idx = range.idxStart; idx != range.getEnd(); idx++) {
    IndexT sIdx = smNonterm.sampleIndex[idx];
      // Branch sense indexing is sample-relative.
    bool sense = branchSense.senseTrue(sIdx, implicitTrue);
    IndexT smIdx = sense ? destTrue++ : destFalse++;
    smNext.sampleIndex[smIdx] = sIdx; // Restages sample index.
    interLevel->rootSuccessor(sIdx, iSet.getPathSucc(sense), smIdx);
  }
}


void ObsFrontier::updateExtinct(const IndexSet& iSet,
				const SampleMap& smNonterm,
				SampleMap& smTerminal) {
  IndexT* destOut = smTerminal.getWriteStart(iSet.getIdxNext());
  IndexRange range = smNonterm.range[iSet.getSplitIdx()];
  for (IndexT idx = range.idxStart; idx != range.getEnd(); idx++) {
    IndexT sIdx = smNonterm.sampleIndex[idx];
    *destOut++ = sIdx;
    interLevel->rootExtinct(sIdx);
  }
}
