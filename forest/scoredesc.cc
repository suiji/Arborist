
#include "scoredesc.h"
#include "forest.h"
#include "quant.h"


unique_ptr<ForestPredictionReg> ScoreDesc::makePredictionReg(const Forest* forest,
							    const Sampler* sampler,
							    size_t nObs) const {
  return make_unique<ForestPredictionReg>(this, sampler, nObs, forest);
}


unique_ptr<ForestPredictionCtg> ScoreDesc::makePredictionCtg(const Forest* forest,
							  const Sampler* sampler,
							  size_t nObs) const {
  return make_unique<ForestPredictionCtg>(this, sampler, nObs, forest);
}
