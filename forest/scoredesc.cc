
#include "scoredesc.h"
#include "forestscorer.h"
#include "quant.h"
//#include "response.h"
//#include "predict.h"
//#include "train.h"


unique_ptr<ForestScorer> ScoreDesc::makeScorer(const ResponseReg* response,
					       const Forest* forest,
					       const Leaf* leaf,
					       const PredictReg* predict,
					       vector<double> quantile) const {
  return make_unique<ForestScorer>(this, response, forest, leaf, predict, std::move(quantile));
}


unique_ptr<ForestScorer> ScoreDesc::makeScorer(const ResponseCtg* response,
					       size_t nObs,
					       bool doProb) const {
  return make_unique<ForestScorer>(this, response, nObs, doProb);
}
