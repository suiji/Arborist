
#include "scoredesc.h"
#include "predictscorer.h"
#include "sampler.h"
#include "predict.h"
#include "train.h"


unique_ptr<PredictScorer> ScoreDesc::makePredictScorer(const Sampler* sampler,
						       const Predict* predict) const {
  return make_unique<PredictScorer>(this, sampler, predict);
}
