// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file dectree.cc

   @brief Methods for building and walking a decision tree.

   @author Mark Seligman
 */


#include "dectree.h"
#include "forest.h"


DecTree::DecTree(const vector<DecNode>& decNode_,
		 const BV& facSplit_,
		 const BV& facObserved_,
		 const vector<double>& nodeScore_) :
  decNode(decNode_),
  facSplit(facSplit_),
  facObserved(facObserved_),
  nodeScore(nodeScore_) {
}


void DecTree::setObsWalker(PredictorT nPredNum) {
  if (facSplit.isEmpty())
    obsWalker = &DecTree::obsNum;
  else if (nPredNum == 0)
    obsWalker = &DecTree::obsFac;
  else
    obsWalker = &DecTree::obsMixed;
}


DecTree::~DecTree() = default;


IndexT DecTree::obsNum(const Forest* forest,
		       size_t obsIdx) {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(forest->baseNum(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT DecTree::obsFac(const Forest* forest,
		       size_t obsIdx) {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(facSplit, facObserved, forest->baseFac(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT DecTree::obsMixed(const Forest* forest,
			 size_t obsIdx) {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(forest, facSplit, facObserved, forest->baseFac(obsIdx), forest->baseNum(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}

