
#include "critencoding.h"
#include "splitfrontier.h"
#include "splitnux.h"


CritEncoding::CritEncoding(const SplitFrontier* sf, const SplitNux& nux, bool incr) :
  sum(0.0), sCount(0), extent(0), scCtg(vector<SumCount>(sf->getNCtg())), implicitTrue(sf->getImplicitTrue(&nux)), increment(incr), exclusive(sf->getCompound()), style(sf->getEncodingStyle()) {
}


IndexT CritEncoding::getSCountTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? sCount : (nux->getSCount() - sCount); 
}


double CritEncoding::getSumTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? sum : (nux->getSum() - sum);
}


IndexT CritEncoding::getExtentTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? extent : (implicitTrue + nux->getExtent() - extent);
}


void CritEncoding::getISetVals(const SplitNux& nux,
			       IndexT& sCountTrue,
			       double& sumTrue,
			       IndexT& extentTrue) const {
  style == EncodingStyle::direct ? accumDirect(sCountTrue, sumTrue, extentTrue) : accumTrue(&nux, sCountTrue, sumTrue, extentTrue);
}


void CritEncoding::accumDirect(IndexT& sCountTrue,
			      double& sumTrue,
			      IndexT& extentTrue) const {
  if (increment) {
    sCountTrue += sCount;
    extentTrue += extent;
    sumTrue += sum;
  }
  else {
    sCountTrue -= sCount;
    extentTrue -= extent;
    sumTrue -= sum;
  }
}


void CritEncoding::accumTrue(const SplitNux* nux,
			     IndexT& sCountTrue,
			     double& sumTrue,
			     IndexT& extentTrue) const {
  if (increment) {
    sCountTrue += getSCountTrue(nux);
    sumTrue += getSumTrue(nux);
    extentTrue += getExtentTrue(nux);
  }
  else {
    sCountTrue -= getSCountTrue(nux);
    sumTrue -= getSumTrue(nux);
    extentTrue -= getExtentTrue(nux);
  }
}
