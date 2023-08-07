#include "predictscorer.h"
#include "scoredesc.h"
#include "sampler.h"
#include "response.h" // Temporary solution.
#include "predict.h"


PredictScorer::PredictScorer(const ScoreDesc* scoreDesc,
	       const Sampler* sampler,
	       const Predict* predict_) :
  nu(scoreDesc->nu),
  baseScore(scoreDesc->baseScore),
  nCtg(sampler->getNCtg()),
  defaultPrediction(sampler->getResponse()->getDefaultPrediction()), // Temporary solution.
  predict(predict_) {
}


double PredictScorer::predictMean(size_t obsIdx) const {
  double sumScore = 0.0;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }
  return nEst > 0 ? sumScore / nEst : defaultPrediction;
}


double PredictScorer::predictSum(size_t obsIdx) const {
  double sumScore = baseScore;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      sumScore += nu * score;
    }
  }
  return sumScore;
}


CtgT PredictScorer::predictProb(size_t obsIdx,
				CtgProb* ctgProb,
				unsigned int census[]) const {
  double logOdds = predictSum(obsIdx);
  double p1 = 1.0 / (1.0 + exp(-logOdds));
  ctgProb->assignBinary(obsIdx, p1);
  CtgT ctg = p1 > 0.5 ? 1 : 0;
  census[ctg] = 1;

  return ctg;
}


PredictorT PredictScorer::predictPlurality(size_t obsIdx, unsigned int* census) const {
  unsigned int nEst = 0; // # participating trees.
  vector<double> ctgJitter(nCtg); // Accumulates jitter by category.
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      PredictorT ctg = floor(score); // Truncates jittered score ut index.
      census[ctg]++;
      ctgJitter[ctg] += score - ctg;
    }
  }
  if (nEst == 0) { // Default category unity, all others zero.
    census[CtgT(defaultPrediction)] = 1;
  }

  return argMaxJitter(census, ctgJitter);
}


PredictorT PredictScorer::argMaxJitter(const unsigned int* census,
				const vector<double>& ctgJitter) const {
  PredictorT argMax = 0;
  IndexT countMax = 0;
  // Assumes at least one slot has nonzero count.
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    IndexT count = census[ctg];
    if (count == 0)
      continue;
    else if (count > countMax) {
      countMax = count;
      argMax = ctg;
    }
    else if (count == countMax) {
      if (ctgJitter[ctg] > ctgJitter[argMax]) {
	argMax = ctg;
      }
    }
  }
  return argMax;
}
