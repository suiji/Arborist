// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file restage.cc

   @brief Methods to update the per-predictor ordering of sampled values following splitting.

   Restaging is implemented by stable partition, directed by sample-indexed predicates.  The predicates are node-specific, as nodes are completely characterized by the samples they index. Sample-to-rank mappings vary by predictor, however, so each node/predictor pair is repartitioned separately using the node's predicate.

   @author Mark Seligman
 */

#include "bv.h"
#include "samplepred.h"
#include "restage.h"
#include "index.h"
#include "splitpred.h"

//include <iostream>
using namespace std;


int RestageMap::nPred = 0;

/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
RestageMap::RestageMap(SplitPred *_splitPred, unsigned int _bagCount, int _splitPrev, int _splitNext) : splitPrev(_splitPrev), splitNext(_splitNext), splitPred(_splitPred) {
  mapNode = new MapNode[splitPrev];
  sIdxLH = new BV(_bagCount);
  sIdxRH = new BV(_bagCount);
}


/**
 */
void RestageMap::Immutables(int _nPred) {
  nPred = _nPred;
}


/**
 */
void RestageMap::DeImmutables() {
  nPred = 0;
}


/**
   @brief Class finalizer.

   @return void.
 */
RestageMap::~RestageMap() {
  delete [] mapNode;
  delete sIdxLH;
  delete sIdxRH;
}


/**
   @brief Consumes all fields in current NodeCache item relevant to restaging.

   @param _splitIdx is the split index.

   @param _lNext is the index node offset of the LHS in the next level.

   @param _rNext is the index node offset of the RHS in the next level.

   @param _lhIdxCount is the count of indices associated with the split's LHS.

   @param _rhIdxCount is the count of indices associated with the split's RHS.

   @return void.
*/
void RestageMap::ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount, int _startIdx, int _endIdx) {
  mapNode[_splitIdx].Init(_splitIdx, _lNext, _rNext, _lhIdxCount, _rhIdxCount, _startIdx, _endIdx);
}

/**
  @brief Finishes setting of map fields.

  @param index caches state information for the predicate bits.

  @param _splitPrev is the number of splitable nodes in the previous level.

  @return void.
*/
void RestageMap::Conclude(const Index *index) {
  endPrev = mapNode[splitPrev-1].EndIdx(); // Terminus of PREVIOUS level.

  int rhIdxTot, lhIdxTot;
  index->PredicateBits(sIdxLH, sIdxRH, lhIdxTot, rhIdxTot);
  rhIdxNext = lhIdxTot;
  endThis = rhIdxTot + lhIdxTot - 1;

  int lhIdx = 0;
  int rhIdx = rhIdxNext;
  for (int splitIdx = 0; splitIdx < splitPrev; splitIdx++) {
    mapNode[splitIdx].UpdateIndices(lhIdx, rhIdx);
  }
}


/**
   @brief Restages predictors and splits as pairs with equal priority.

   @param samplePred holds the restaging area.

   @param level is the next level to be split.

   @return void, with side-effected restaging buffers.
 */
void RestageMap::RestageLevel(SamplePred *samplePred, unsigned int level) {
  int predIdx, splitIdx;
  SPNode *source, *targ;
  unsigned int *sIdxSource, *sIdxTarg;

#pragma omp parallel default(shared) private(predIdx, splitIdx, source, sIdxSource, targ, sIdxTarg)
  {
#pragma omp for schedule(dynamic, 1) collapse(2)
    for (predIdx = 0; predIdx < nPred; predIdx++) {
      for (splitIdx = 0; splitIdx < splitPrev; splitIdx++) {
        samplePred->Buffers(predIdx, level, source, sIdxSource, targ, sIdxTarg);
        if (!splitPred->Singleton(splitIdx, predIdx)) {
          mapNode[splitIdx].Restage(source, sIdxSource, targ, sIdxTarg, sIdxLH, sIdxRH);
          mapNode[splitIdx].Singletons(splitPred, targ, predIdx);
        }
      }
    }
  }
}


/**
   @brief Advises SplitPred of any singletons arising as a result of this
   restaging.

   @param splitPred is the current SplitPred object.

   @param targ is the restaged data.

   @param predIdx is the predictor index.

   @param lhIdx is the starting index of the left successor.

   @param rhIdx is the starting index of the right successor.

   @return void.
 */
void MapNode::Singletons(SplitPred *splitPred, const SPNode targ[], int predIdx) {
  if (lNext >= 0 && targ->IsRun(idxNextL, idxNextL + lhIdxCount - 1)) {
      splitPred->LengthNext(lNext, predIdx) = 1;
  }
  if (rNext >= 0 && targ->IsRun(idxNextR, idxNextR + rhIdxCount - 1)) {
      splitPred->LengthNext(rNext, predIdx) = 1;
  }
}


/**
   @brief Assigns and accumulates left, right starting indices.

   @param lhIdx inputs the left index for this node and outputs the
   left index for the next node.

   @param rhIdx inputs the right index for this node and outputs the
   right index for the next node.

  @return void.
 */
void MapNode::UpdateIndices(int &lhIdx, int &rhIdx) {
  idxNextL = lhIdx;
  idxNextR = rhIdx;
  lhIdx += (lNext >= 0 ? lhIdxCount : 0);
  rhIdx += (rNext >= 0 ? rhIdxCount : 0);
}


/**
   @brief Sends contents of previous level's SamplePreds to this level's descendents, via a stable partition.

   @param source contains the previous level's SamplePreds.

   @param targ outputs this level's SamplePreds.

   @param lhIdx is the index node offset for the LHS.

   @param rhIdx is the index node offset for the RHS.

   @return void, with output parameter vector.
*/
void MapNode::Restage(const SPNode source[], const unsigned int sIdxSource[], SPNode targ[], unsigned int sIdxTarg[], const BV *sIdxLH, const BV *sIdxRH) {
  if (lNext >= 0 && rNext >= 0) // Both subnodes nonterminal.
    RestageLR(source, sIdxSource, targ, sIdxTarg, startIdx, endIdx, sIdxLH, idxNextL, idxNextR);
  else if (lNext >= 0) // Only LH subnode nonterminal.
    RestageSingle(source, sIdxSource, targ, sIdxTarg, startIdx, endIdx, sIdxLH, idxNextL);
  else if (rNext >= 0) // Only RH subnode nonterminal.
    RestageSingle(source, sIdxSource, targ, sIdxTarg, startIdx, endIdx, sIdxRH, idxNextR);

  // Otherwise, either node is itself terminal or both subnodes are.
  
  // Post-conditions:  lhIdx = lhIdx in + lhIdxCount && rhIdx = rhIdx in + rhIdxCount
}

/**
   @brief Sends SamplePred contents to both LH and RH targets.

   @param source contains the previous level's SamplePred values.

   @param targ outputs the current level's SamplePred values.

   @param startIdx is the first index in the node being restaged.

   @param endIdx is the last index in the node being restaged.

   @param lhIdx is the index node offset of the LHS.

   @param rhIdx is the index node offset of the RHS.

   @return void.
 */
// Target nodes should all equal either lh or rh.
//
void MapNode::RestageLR(const SPNode source[], const unsigned int sIdxSource[], SPNode targ[], unsigned int sIdxTarg[], int startIdx, int endIdx, const BV *bvL, int lhIdx, int rhIdx) {
  for (int i = startIdx; i <= endIdx; i++) {
    unsigned int sIdx = sIdxSource[i];
    int destIdx = bvL->IsSet(sIdx) ? lhIdx++ : rhIdx++;
    sIdxTarg[destIdx] = sIdx;
    targ[destIdx] = source[i];
  }
}

/**
   @brief Sends SamplePred contents to one of either LH or RH targets.

   @param source contains the previous level's SamplePred values.

   @param targ outputs the current level's SamplePred values.

   @param startIdx is the first index in the node being restaged.

   @param endIdx is the last index in the node being restaged.

   @param bv is the bit vector testing the given handedness.

   @param idx is the offset of the descendent index node.

   @return void.
 */
// Target nodes should all be either leaf or set in bv[].
void MapNode::RestageSingle(const SPNode source[], const unsigned int sIdxSource[], SPNode targ[], unsigned int sIdxTarg[], int startIdx, int endIdx, const BV *bv, int idx) {
  for (int i = startIdx; i <= endIdx; i++) {
    unsigned int sIdx = sIdxSource[i];
    if (bv->IsSet(sIdx)) {
      int destIdx = idx++;
      sIdxTarg[destIdx] = sIdx;
      targ[destIdx] = source[i];
    }
  }
}
