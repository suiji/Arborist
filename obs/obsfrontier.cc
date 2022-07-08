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
#include "layout.h"
#include "frontier.h"
#include "obsfrontier.h"
#include "path.h"
#include "interlevel.h"
#include "partition.h"
#include "samplemap.h"
#include "sampleobs.h"
#include "indexset.h"
#include "algparam.h"


ObsFrontier::ObsFrontier(const Frontier* frontier_,
			 InterLevel* interLevel_) :
  frontier(frontier_),
  interLevel(interLevel_),
  nPred(interLevel->getNPred()),
  nSplit(interLevel->getNSplit()),
  //  noIndex(frontier->getNonterminalEnd()),
  node2Front(vector<IndexRange>(nSplit)), // Initialized to empty.
  stagedCell(vector<vector<StagedCell>>(nSplit)),
  stageCount(0),
  rankOffset(0),
  layerIdx(0), // Not on layer yet, however.
  nodePath(backScale(nSplit)) {
  NodePath::setNoSplit(frontier->getBagCount());
  // Coprocessor only.
  // LiveBits df;
  //  fill(mrra.begin(), mrra.end(), df);
}


void ObsFrontier::prestageRoot() {
  for (PredictorT predIdx = 0; predIdx != nPred; predIdx++) {
    interLevel->setStaged(0, predIdx, predIdx);
    stagedCell[0].emplace_back(predIdx, frontier->getBagCount(), rankOffset);
    rankOffset += frontier->getBagCount(); // TODO:  sharpen.
  }
  stageCount = nPred;
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
    stagedCell[nodeIdx].emplace_back(nodeIdx, cell, frontier->getNodeRange(nodeIdx), rankOffset);
    rankOffset += min(frontier->getExtent(nodeIdx), cell.getRankCount());
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


void ObsFrontier::setRankTarget() {
  rankTarget = vector<IndexT>(rankOffset);
}


double ObsFrontier::interpolateBackRank(const SplitNux* cand,
					IndexT backIdxL,
					IndexT backIdxR) const {
  const StagedCell* cell = cand->getStagedCell();
  IndexT rankLeft = rankTarget[cell->rankRear(backIdxL)];
  IndexT rankRight = rankTarget[cell->rankRear(backIdxR)];
  IndexRange rankRange(rankLeft, rankRight - rankLeft);

  return rankRange.interpolate(cand->getSplitQuant());
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
				const vector<IndexSet>& frontierNext) {
  front2Node = vector<IndexT>(frontierNext.size());
  IndexT terminalCount = 0;
  for (IndexT parIdx = 0; parIdx < frontierNodes.size(); parIdx++) {
    if (frontierNodes[parIdx].isTerminal()) {
      terminalCount++;
      delistNode(parIdx);
    }
    else {
      setFrontRange(frontierNext, parIdx, IndexRange(2 * (parIdx - terminalCount), 2));
    }
  }
}


void ObsFrontier::setFrontRange(const vector<IndexSet>& frontierNext,
				IndexT nodeIdx,
				const IndexRange& range) {
  node2Front[nodeIdx] = range;
  NodePath* pathBase = &nodePath[backScale(nodeIdx)];
  for (IndexT frontIdx = range.getStart(); frontIdx != range.getEnd(); frontIdx++) {
    pathInit(pathBase, frontierNext[frontIdx]);
    front2Node[frontIdx] = nodeIdx;
  }
}


void ObsFrontier::pathInit(NodePath* pathBase, const IndexSet& iSet) {
  pathBase[iSet.getPath(pathMask())].init(frontier, iSet);
}


void ObsFrontier::applyFront(const ObsFrontier* ofFront,
			    const vector<IndexSet>& frontierNext) {
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
      setFrontRange(frontierNext, nodeIdx, frontRange);
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
				const Layout* layout,
				const SampleObs* sampleObs) {
  StagedCell& cell = stagedCell[0][predIdx];
  obsPart->setStageRange(predIdx, layout->getSafeRange(predIdx, frontier->getBagCount()));
  IndexT rankDense = layout->getDenseRank(predIdx);
  IndexT* idxStart;
  Obs* srStart = obsPart->buffers(predIdx, 0, idxStart);
  Obs* spn = srStart;
  IndexT* sIdx = idxStart;
  IndexT rankPrev = interLevel->getNoRank();
  IndexT* rankTarg = &rankTarget[cell.rankStart];
  for (auto rle : layout->getRLE(predIdx)) {
    IndexT rank = rle.val;
    if (rank != rankDense) {
      for (IndexT row = rle.row; row < rle.row + rle.extent; row++) {
	IndexT smpIdx;
	SampleNux sampleNux;
	if (sampleObs->isSampled(row, smpIdx, sampleNux)) {
	  bool tie = rank == rankPrev;
	  spn++->join(sampleNux, tie);
	  *sIdx++ = smpIdx;
	  if (!tie) {
	    *rankTarg++ = rank;
	    rankPrev = rank;
	  }
	}
      }
    }
    else {
      cell.preResidual = spn - srStart;
    }
  }
  cell.updateRange(frontier->getBagCount() - (spn - srStart));
  if (cell.implicitObs())
    *rankTarg++ = rankDense;
  cell.setRankCount(rankTarg - &rankTarget[cell.rankStart]);

  if (cell.isSingleton()) {
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

  vector<IndexT> rankScatter(backScale(1));
  vector<IndexT> obsScatter =  packTargets(obsPart, mrra, tcp, rankScatter);
  obsRestage(obsPart, rankScatter, mrra, obsScatter, ofFront->getRankTarget());

  unsigned int nSingleton = 0;

  // Speculatively assumes mrra has residual:
  IndexT residualRank = rankTarget[mrra.residualPosition()];
  for (PathT path = 0; path != backScale(1); path++) {
    StagedCell* cell = tcp[path];
    if (cell != nullptr) {
      if (cell->implicitObs()) { // Only has residual if mrra does.
	ofFront->setRank(rankScatter[path]++, residualRank);
      }
      cell->setRankCount(rankScatter[path] - cell->rankStart);
      if (cell->isSingleton()) {
	interLevel->delist(cell->coord);
	cell->delist();
	nSingleton++;
      }
    }
  }

  return nSingleton;
}


// Successors may or may not themselves be dense.
vector<IndexT> ObsFrontier::packTargets(ObsPart* obsPart,
					const StagedCell& mrra,
					vector<StagedCell*>& tcp,
					vector<IndexT>& rankScatter) const {
  vector<IndexT> preResidual(backScale(1));
  vector<IndexT> pathCount = pathRestage(obsPart, preResidual, mrra);
  vector<IndexT> obsScatter(backScale(1));
  IndexT idxStart = mrra.obsRange.getStart();
  const NodePath* pathPos = &nodePath[backScale(mrra.getNodeIdx())];
  PredictorT predIdx = mrra.getPredIdx();
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexT frontIdx;
    if (pathPos[path].getFrontIdx(frontIdx)) {
      IndexT extentDense = pathCount[path];
      tcp[path] = interLevel->getFrontCellAddr(SplitCoord(frontIdx, predIdx));
      tcp[path]->setRange(idxStart, extentDense);
      tcp[path]->setPreresidual(preResidual[path]);
      obsScatter[path] = idxStart;
      rankScatter[path] = tcp[path]->rankStart;
      idxStart += extentDense;
    }
  }
  return obsScatter;
}


vector<IndexT> ObsFrontier::pathRestage(ObsPart* obsPart,
					vector<IndexT>& preResidual,
					const StagedCell& mrra) const {
  IndexRange idxRange = mrra.obsRange;
  const IdxPath* idxPath = interLevel->getRootPath();
  const IndexT* indexVec = obsPart->idxBuffer(&mrra);
  PathT* prePath = interLevel->getPathBlock(mrra.getPredIdx());
  vector<IndexT> pathCount(backScale(1));

  // Loop can be streamlined if mrra has no implicit observations
  // or if splitting will not be cut-based:
  IndexT preResidualThis = mrra.preResidual;
  bool cutSeen = mrra.implicitObs() ? false : true;
  for (IndexT idx = idxRange.getStart(); idx != idxRange.getEnd(); idx++) {
    cutSeen = cutSeen || ((idx - idxRange.getStart()) >= preResidualThis);
    PathT path;
    if (idxPath->pathSucc(indexVec[idx], pathMask(), path)) {
      pathCount[path]++;
      if (!cutSeen)
	preResidual[path]++;
    }
    prePath[idx] = path;
  }

  return pathCount;
}


void ObsFrontier::obsRestage(ObsPart* obsPart,
			     vector<IndexT>& rankScatter,
			     const StagedCell& mrra,
			     vector<IndexT>& obsScatter,
			     vector<IndexT>& ranks) const {
  const PathT* prePath = interLevel->getPathBlock(mrra.getPredIdx());

  Obs *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  obsPart->buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  vector<IndexT> rankPrev(rankScatter.size());
  fill(rankPrev.begin(), rankPrev.end(), interLevel->getNoRank());
  IndexT rankIdx = mrra.rankStart;
  if (mrra.hasTies()) {
    srSource[mrra.obsRange.getStart()].setTie(true); // Fillip;  temporary.
    for (IndexT idx = mrra.obsRange.getStart(); idx < mrra.obsRange.getEnd(); idx++) {
      Obs sourceNode = srSource[idx];
      rankIdx += sourceNode.isTied() ? 0 : 1;
      PathT path = prePath[idx];
      if (NodePath::isActive(path)) {
	IndexT rank = rankTarget[rankIdx];
	if (rank != rankPrev[path]) {
	  sourceNode.setTie(false);
	  rankPrev[path] = rank;
	  IndexT rankDest = rankScatter[path]++;
	  ranks[rankDest] = rank;
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
  else {
    // Loop can be streamlined if no ties present:
    for (IndexT idx = mrra.obsRange.getStart(); idx < mrra.obsRange.getEnd(); idx++) {
      PathT path = prePath[idx];
      if (NodePath::isActive(path)) {
	IndexT rank = rankTarget[rankIdx];
	if (rank != rankPrev[path]) {
	  rankPrev[path] = rank;
	  IndexT rankDest = rankScatter[path]++;
	  ranks[rankDest] = rank;
	}
	IndexT obsDest = obsScatter[path]++;
	srTarg[obsDest] = srSource[idx];
	idxTarg[obsDest] = idxSource[idx];
      }
      rankIdx++;
    }
  }  
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
