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
			 IndexT nSplit_,
			 PredictorT nPred_,
			 IndexT idxLive,
			 InterLevel* interLevel_) :
  frontier(frontier_),
  interLevel(interLevel_),
  nPred(nPred_),
  nSplit(nSplit_),
  //  noIndex(frontier->getBagCount()),
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
  StagedCell* sc = interLevel->getFrontCellAddr(SplitCoord(0, predIdx));
  obsPart->setStageRange(predIdx, layout->getSafeRange(predIdx, frontier->getBagCount()));
  IndexT rankDense = layout->getDenseRank(predIdx);
  IndexT* idxStart;
  Obs* srStart = obsPart->buffers(sc->getPredIdx(), 0, idxStart);
  Obs* spn = srStart;
  IndexT* sIdx = idxStart;
  IndexT rankPrev = 0xffffffff; // frame->getNRow();
  IndexT* rankTarg = &rankTarget[sc->rankStart];
  for (auto rle : layout->getRLE(predIdx)) {
    IndexT rank = rle.val;
    if (rank != rankDense) {
      for (IndexT row = rle.row; row < rle.row + rle.extent; row++) {
	IndexT smpIdx;
	SampleNux sampleNux;
	if (sampleObs->isSampled(row, smpIdx, sampleNux)) {
	  *sIdx++ = smpIdx;
	  spn++->join(sampleNux, rank);
	  if (rank != rankPrev) {
	    *rankTarg++ = rank;
	    rankPrev = rank;
	  }
	}
      }
    }
    else {
      sc->denseCut = spn - srStart;
    }
  }
  sc->updateRange(frontier->getBagCount() - (spn - srStart));
  sc->setRankCount(rankTarg - &rankTarget[sc->rankStart]);
  if (sc->isSingleton()) {
    interLevel->delist(sc->coord);
    sc->delist();
    return 1;
  }
  else return 0;
}


unsigned int ObsFrontier::restage(ObsPart* obsPart,
				  const StagedCell& mrra,
				  ObsFrontier* ofFront) {
  vector<StagedCell*> tcp(backScale(1));
  fill(tcp.begin(), tcp.end(), nullptr);

  vector<IndexT> rankScatter(backScale(1));
  vector<IndexT> obsScatter =  packTargets(obsPart, mrra, tcp, rankScatter);
  ofFront->obsRestage(obsPart, rankScatter, mrra, obsScatter);

  unsigned int nSingleton = 0;
  for (PathT path = 0; path != backScale(1); path++) {
    StagedCell* cell = tcp[path];
    if (cell != nullptr) {
      cell->setRankCount(rankScatter[path] - tcp[path]->rankStart);
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
					vector<IndexT>& rankScatter) {
  vector<IndexT> pathCount = pathRestage(obsPart, mrra);
  vector<IndexT> obsScatter(backScale(1));
  IndexT idxStart = mrra.range.getStart();
  const NodePath* pathPos = &nodePath[backScale(mrra.getNodeIdx())];
  PredictorT predIdx = mrra.getPredIdx();
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexT frontIdx;
    if (pathPos[path].getFrontIdx(frontIdx)) {
      IndexT extentDense = pathCount[path];
      tcp[path] = interLevel->getFrontCellAddr(SplitCoord(frontIdx, predIdx));
      tcp[path]->setRange(idxStart, extentDense);
      obsScatter[path] = idxStart;
      rankScatter[path] = tcp[path]->rankStart;
      idxStart += extentDense;
    }
  }
  return obsScatter;
}


vector<IndexT> ObsFrontier::pathRestage(ObsPart* obsPart,
					const StagedCell& mrra) {
  IndexRange idxRange = mrra.range;
  IdxPath* idxPath = getIndexPath();
  IndexT* indexVec = obsPart->idxBuffer(&mrra);
  PathT* prePath = obsPart->getPathBlock(mrra.getPredIdx());
  vector<IndexT> pathCount(backScale(1));
  for (IndexT idx = idxRange.getStart(); idx != idxRange.getEnd(); idx++) {
    PathT path;
    if (idxPath->pathSucc(indexVec[idx], pathMask(), path)) {
      pathCount[path]++;
    }
    prePath[idx] = path;
  }

  return pathCount;
}


void ObsFrontier::obsRestage(ObsPart* obsPart,
			     vector<IndexT>& rankScatter,
			     const StagedCell& mrra,
			     vector<IndexT>& obsScatter) {
  Obs *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  obsPart->buffers(&mrra, srSource, idxSource, srTarg, idxTarg);
  const PathT* pathBlock = obsPart->getPathBlock(mrra.getPredIdx());
  vector<IndexT> rankPrev(rankScatter.size());
  fill(rankPrev.begin(), rankPrev.end(), 0xffffffff); // frame->getNRow()
  
  IndexRange idxRange = mrra.range;
  for (IndexT idx = idxRange.idxStart; idx < idxRange.getEnd(); idx++) {
    unsigned int path = pathBlock[idx];
    if (NodePath::isActive(path)) {
      Obs sourceNode = srSource[idx];
      IndexT obsDest = obsScatter[path]++;
      srTarg[obsDest] = sourceNode;
      idxTarg[obsDest] = idxSource[idx];
      IndexT rank = sourceNode.getRank();
      if (rank != rankPrev[path]) {
	rankPrev[path] = rank;
	IndexT rankDest = rankScatter[path]++;
	rankTarget[rankDest] = rank;
      }
    }
  }
}


IdxPath* ObsFrontier::getIndexPath() const {
  return interLevel->getRootPath();
}


void ObsFrontier::updateMap(const IndexSet& iSet,
			    const BranchSense* branchSense,
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


void ObsFrontier::updateLive(const BranchSense* branchSense,
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
    bool sense = branchSense->senseTrue(sIdx, implicitTrue);
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
