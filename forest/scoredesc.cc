
#include "scoredesc.h"
#include "predict.h"
#include "quant.h"


unique_ptr<ForestPredictionReg> ScoreDesc::makePredictionReg(const Predict* predict,
							     const Sampler* sampler,
							     bool reportAuxiliary) const {
  return make_unique<ForestPredictionReg>(this, sampler, predict, reportAuxiliary);
}


unique_ptr<ForestPredictionCtg> ScoreDesc::makePredictionCtg(const Predict* predict,
							     const Sampler* sampler,
							     bool reportAuxiliary) const {
  return make_unique<ForestPredictionCtg>(this, sampler, predict, reportAuxiliary);
}
