
#include "critencoding.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "branchsense.h"
#include "obspart.h"


CritEncoding::CritEncoding(const SplitFrontier* sf, const SplitNux& nux_, bool incr) :
  sum(0.0), sCount(0), extent(0), nux(nux_), scCtg(vector<SumCount>(sf->getNCtg())), implicitTrue(sf->getImplicitTrue(&nux)), increment(incr), exclusive(sf->getCompound()), style(sf->getEncodingStyle()) {
}


IndexT CritEncoding::getSCountTrue() const {
  return implicitTrue == 0 ? sCount : (nux.getSCount() - sCount); 
}


double CritEncoding::getSumTrue() const {
  return implicitTrue == 0 ? sum : (nux.getSum() - sum);
}


IndexT CritEncoding::getExtentTrue() const {
  return implicitTrue == 0 ? extent : (implicitTrue + nux.getExtent() - extent);
}


void CritEncoding::getISetVals(IndexT& sCountTrue,
			       double& sumTrue,
			       IndexT& extentTrue) const {
  style == EncodingStyle::direct ? accumDirect(sCountTrue, sumTrue, extentTrue) : accumTrue(sCountTrue, sumTrue, extentTrue);
}


void CritEncoding::accumDirect(IndexT& sCountTrue,
			      double& sumTrue,
			      IndexT& extentTrue) const {
  int coeff = increment ? 1 : -1;
  sCountTrue += coeff * sCount;
  extentTrue += coeff * extent;
  sumTrue += coeff * sum;
}


void CritEncoding::accumTrue(IndexT& sCountTrue,
			     double& sumTrue,
			     IndexT& extentTrue) const {
  if (increment) {
    sCountTrue += getSCountTrue();
    sumTrue += getSumTrue();
    extentTrue += getExtentTrue();
  }
  else {
    sCountTrue -= getSCountTrue();
    sumTrue -= getSumTrue();
    extentTrue -= getExtentTrue();
  }
}


void CritEncoding::branchUpdate(const SplitFrontier* sf,
				const IndexRange& range,
				BranchSense* branchSense) {
  if (!range.empty()) {
    branchUpdate(sf->getPartition(), range, branchSense);
  }
  else {
    for (auto rg : sf->getRange(nux, *this)) {
      branchUpdate(sf->getPartition(), rg, branchSense);
    }
  }
}


void CritEncoding::branchUpdate(const ObsPart* obsPart,
				const IndexRange& range,
				BranchSense* branchSense) {
  IndexT* sIdx;
  SampleRank* spn = obsPart->getBuffers(nux, sIdx);
  if (increment) {
    branchSet(sIdx, spn, range, branchSense);
  }
  else {
    branchUnset(sIdx, spn, range, branchSense);
  }
}


void CritEncoding::branchSet(IndexT* sIdx,
			     SampleRank* spn,
			     const IndexRange& range,
			     BranchSense *branchSense) {
  // Encodes iff explicit state has been reset.
  if (exclusive) {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      if (!branchSense->isExplicit(sIdx[opIdx])) {
	branchSense->set(sIdx[opIdx], trueEncoding());
	encode(spn[opIdx]);
      }
    }
  }
  else {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      branchSense->set(sIdx[opIdx], trueEncoding());
      encode(spn[opIdx]);
    }
  }
}


void CritEncoding::branchUnset(IndexT* sIdx,
			       SampleRank* spn,
			       const IndexRange& range,
			       BranchSense* branchSense) {
  if (exclusive) {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      if (branchSense->isExplicit(sIdx[opIdx])) {
	branchSense->unset(sIdx[opIdx], trueEncoding());
	encode(spn[opIdx]);
      }
    }
  }
  else {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      branchSense->unset(sIdx[opIdx], trueEncoding());
      encode(spn[opIdx]);
    }
  }
}


void CritEncoding::encode(const SampleRank& obs) {
  accum(obs.getSum(), obs.getSCount(), obs.getCtg());
}
