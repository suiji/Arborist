
#include "train.h"
#include "predictorframe.h"
#include "forest.h"
#include "sampler.h"
#include "nodescorer.h"
#include "booster.h"


Train::Train(const PredictorFrame* frame,
	     const Sampler* sampler,
	     Forest* forest_) :
  predInfo(vector<double>(frame->getNPred())),
  forest(forest_),
  nodeScorer(sampler->getNCtg() > 0 ? NodeScorer::makePlurality() : NodeScorer::makeMean()) {
  if (sampler->getNCtg() > 0)
    Booster::setPlurality();
  else
    Booster::setMean();
}

