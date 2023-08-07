
#include "train.h"
#include "predictorframe.h"
#include "forest.h"
#include "sampler.h"
#include "frontierscorer.h"

// Type completion only:
#include "sampledobs.h"


Train::Train(const PredictorFrame* frame,
	     const Sampler* sampler,
	     Forest* forest_) :
  predInfo(vector<double>(frame->getNPred())),
  forest(forest_),
  frontierScorer(sampler->getNCtg() > 0 ? FrontierScorer::makePlurality() : FrontierScorer::makeMean()) {
}

