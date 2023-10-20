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
#include "predictframe.h"


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


IndexT DecTree::obsNum(const PredictFrame& frame,
		       size_t obsIdx) const {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(frame.baseNum(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT DecTree::obsFac(const PredictFrame& frame,
		       size_t obsIdx) const {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(facSplit, facObserved, frame.baseFac(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT DecTree::obsMixed(const PredictFrame& frame,
			 size_t obsIdx) const {
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = decNode[idx].advance(frame, facSplit, facObserved, frame.baseFac(obsIdx), frame.baseNum(obsIdx));
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}

