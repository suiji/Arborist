
#include "cartnode.h"
#include "predictframe.h"
#include "dectree.h"


IndexT CartNode::advance(const PredictFrame* frame,
			 const DecTree* decTree,
			 size_t obsIdx) const {
  if (isTerminal())
    return 0;
  else {
    bool isFactor;
    IndexT blockIdx = frame->getIdx(getPredIdx(), isFactor);
    if (isFactor) {
      const CtgT* obsFac = frame->baseFac(obsIdx);
      return TreeNode::advanceFactor(decTree->getFacSplit(), obsFac[blockIdx] + getBitOffset());
    }
    else {
      const double* obsNum = frame->baseNum(obsIdx);
      return TreeNode::advanceNum(obsNum[blockIdx]);
    }
  }
}


IndexT CartNode::advanceTrap(const PredictFrame* frame,
			     const DecTree* decTree,
			     size_t obsIdx) const {
  if (isTerminal())
    return 0;
  else {
    bool isFactor;
    IndexT blockIdx = frame->getIdx(getPredIdx(), isFactor);
    if (isFactor) {
      const CtgT* obsFac = frame->baseFac(obsIdx);
      return TreeNode::advanceFactorTrap(decTree->getFacSplit(), decTree->getFacObserved(), obsFac[blockIdx] + getBitOffset());
    }
    else {
      const double* obsNum = frame->baseNum(obsIdx);
      return TreeNode::advanceNumTrap(obsNum[blockIdx]);
    }
  }
}
