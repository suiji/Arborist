

#include "scoredesc.h"
#include "sampler.h"
#include "forest.h"
#include "predict.h"
#include "prediction.h"
#include "quant.h"
#include "response.h"

bool ForestPrediction::reportIndices = false;
bool CtgProb::reportProbabilities = false;


map<const string, function<void(ForestPredictionReg*, const Predict*, size_t)>> ForestPredictionReg::scorerTable {
  {"mean", &ForestPredictionReg::predictMean},
  {"sum", &ForestPredictionReg::predictSum}
};


map<const string, function<void(ForestPredictionCtg*, const Predict*, size_t)>> ForestPredictionCtg::scorerTable {
  {"plurality", &ForestPredictionCtg::predictPlurality},
  {"logistic", &ForestPredictionCtg::predictLogistic}
};


ForestPrediction::ForestPrediction(const Predict* predict,
				   const struct ScoreDesc* scoreDesc) :
  baseScore(scoreDesc->baseScore),
  nu(scoreDesc->nu),
  idxFinal(vector<size_t>(reportIndices ? predict->getNTree() * predict->getNObs() : 0)) {
}


void ForestPrediction::cacheIndices(vector<IndexT>& indices,
				   size_t span,
				   size_t obsStart) {
  if (reportIndices)
    copy(&indices[0], &indices[span], &idxFinal[obsStart]);
}


ForestPredictionCtg::ForestPredictionCtg(const ScoreDesc* scoreDesc,
					 const Sampler* sampler,
					 const Predict* predict,
					 bool reportAuxiliary) :
  ForestPrediction(predict, scoreDesc),
  scorer(scorerTable[scoreDesc->scorer]),
  nCtg(sampler->getNCtg()),
  prediction(Prediction<CtgT>(predict->getNObs())),
  defaultPrediction(reinterpret_cast<const ResponseCtg*>(sampler->getResponse())->getDefaultPrediction()),
  census(predict->getNObs() * nCtg),
  ctgProb(make_unique<CtgProb>(sampler, predict->getNObs(), reportAuxiliary)) {
}


ForestPredictionReg::ForestPredictionReg(const ScoreDesc* scoreDesc,
					 const Sampler* sampler,
					 const Predict* predict,
					 bool reportAuxiliary) :
  ForestPrediction(predict, scoreDesc),
  scorer(scorerTable[scoreDesc->scorer]),
  prediction(Prediction<double>(predict->getNObs())),
  defaultPrediction(reinterpret_cast<const ResponseReg*>(sampler->getResponse())->getDefaultPrediction()),
  quant(make_unique<Quant>(sampler, predict, reportAuxiliary)) {
}


void ForestPredictionReg::predictMean(const Predict* predict, size_t obsIdx) {
  double sumScore = 0.0;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx != predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }
  setScore(predict, obsIdx, ScoreCount(nEst, nEst > 0 ? sumScore / nEst : defaultPrediction));
}


void ForestPredictionReg::predictSum(const Predict* predict, size_t obsIdx) {
  double sumScore = baseScore;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx != predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      sumScore += nu * score;
      nEst++;
    }
  }
  setScore(predict, obsIdx, ScoreCount(nEst, sumScore));
}


ScoreCount ForestPredictionCtg::predictLogOdds(const Predict* predict, size_t obsIdx) const {
  double sumScore = baseScore;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx != predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      sumScore += nu * score;
      nEst++;
    }
  }
  return ScoreCount(nEst, sumScore);
}


void ForestPredictionCtg::predictLogistic(const Predict* predict, size_t obsIdx) {
  ScoreCount logOdds = predictLogOdds(predict, obsIdx);
  double p1 = 1.0 / (1.0 + exp(-logOdds.score.num));
  ctgProb->assignBinary(obsIdx, p1); // LOWER
  CtgT ctg = p1 > 0.5 ? 1 : 0;
  census[obsIdx * nCtg + ctg] = 1;
  setScore(obsIdx, ScoreCount(logOdds.nEst, ctg));
}


void ForestPredictionCtg::predictPlurality(const Predict* predict, size_t obsIdx) {
  unsigned int nEst = 0; // # participating trees.
  vector<double> ctgJitter(nCtg); // Accumulates jitter by category.
  unsigned int *censusRow = &census[obsIdx * nCtg];
  for (unsigned int tIdx = 0; tIdx != predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      CtgT ctg = floor(score); // Truncates jittered score ut index.
      censusRow[ctg]++;
      ctgJitter[ctg] += score - ctg;
    }
  }
  vector<double> numVec(nCtg);
  if (nEst == 0) { // Default category unity, all others zero.
    censusRow[CtgT(defaultPrediction)] = 1; // EXIT
    numVec[CtgT(defaultPrediction)] = 1;
  }
  else {
    // Scales predictions ut break ties with minimal effect on
    // probabilities.
    double scale = 1.0 / (2 * nEst);
    for (CtgT ctg = 0; ctg != nCtg; ctg++) {
      numVec[ctg] = censusRow[ctg] + ctgJitter[ctg] * scale;
    }
  }

  ctgProb->predictRow(obsIdx, numVec, nEst); // LOWER
  setScore(obsIdx, ScoreCount(nEst, argMaxJitter(numVec)));
}

CtgT ForestPredictionCtg::argMaxJitter(const vector<double>& numVec) const {
  CtgT argMax = 0;
  double valMax = 0.0;
  // Assumes at least one slot has nonzero count.
  for (CtgT ctg = 0; ctg != nCtg; ctg++) {
    double numVal = numVec[ctg];
    if (numVal > valMax) {
      valMax = numVal;
      argMax = ctg;
    }
  }
  return argMax;
}

void ForestPredictionCtg::setScore(size_t obsIdx, ScoreCount score) {
  prediction.setScore(obsIdx, score.score.ctg);
}


const vector<double>& ForestPredictionCtg::getProb() const {
  return ctgProb->getProb();
}


void ForestPredictionReg::setScore(const Predict* predict, size_t obsIdx, ScoreCount score) {
  prediction.setScore(obsIdx, score.score.num);
  // Relies on score having been assigned:
  quant->predictRow(predict, this, obsIdx);
}


unique_ptr<TestReg> ForestPredictionReg::test(const vector<double>& yTest) const {
  if (yTest.empty())
    return make_unique<TestReg>();

  double absErr = 0.0;
  double SSE = 0.0;

  const vector<double>& yPred = prediction.value;
  for (size_t obsIdx = 0; obsIdx != yTest.size(); obsIdx++) {
    double err = fabs(yTest[obsIdx] - yPred[obsIdx]);
    absErr += err;
    SSE += err * err;
  }

  return make_unique<TestReg>(SSE, absErr);
}



unique_ptr<TestCtg> ForestPredictionCtg::test(const vector<CtgT>& yTest) const {
  if (yTest.empty())
    return make_unique<TestCtg>();

  unique_ptr<TestCtg> testCtg = make_unique<TestCtg>(nCtg, 1 + *max_element(yTest.begin(), yTest.end()));
  testCtg->buildConfusion(yTest, prediction.value);
  return testCtg;
}


void TestCtg::buildConfusion(const vector<CtgT>& yTest,
			     const vector<CtgT>& yPred) {
  for (size_t obsIdx = 0; obsIdx != yTest.size(); obsIdx++) {
    confusion[yTest[obsIdx] * nCtgTrain + yPred[obsIdx]]++;
  }

  setMisprediction(yTest.size());
}


void TestCtg::setMisprediction(size_t nObs) {
  size_t totWrong = 0;
  for (PredictorT ctgRec = 0; ctgRec < nCtgMerged; ctgRec++) {
    size_t numWrong = 0;
    size_t numRight = 0;
    for (PredictorT ctgPred = 0; ctgPred < nCtgTrain; ctgPred++) {
      size_t numConf = confusion[ctgRec * nCtgTrain + ctgPred];
      if (ctgPred != ctgRec) {  // Misprediction iff off-diagonal.
        numWrong += numConf;
      }
      else {
        numRight = numConf;
      }
    }
    totWrong += numWrong;
    
    misprediction[ctgRec] = numWrong + numRight == 0 ? 0.0 : double(numWrong) / double(numWrong + numRight);
  }
  oobErr = double(totWrong) / nObs;
}


vector<vector<double>> TestReg::getSSEPermuted(const vector<vector<unique_ptr<TestReg>>>& testPerm) {
  unsigned int nPred = testPerm.size();
  unsigned int nPerm = testPerm[0].size();
  vector<vector<double>> ssePerm(nPred);

  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    ssePerm[predIdx] = vector<double>(nPerm);
    for (unsigned int permIdx = 0; permIdx != nPerm; permIdx++) {
      ssePerm[predIdx][permIdx] = testPerm[predIdx][permIdx]->SSE;
    }
  }

  return ssePerm;
}


vector<vector<double>> TestReg::getSAEPermuted(const vector<vector<unique_ptr<TestReg>>>& testPerm) {
  unsigned int nPred = testPerm.size();
  unsigned int nPerm = testPerm[0].size();
  vector<vector<double>> saePerm(nPred);

  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    saePerm[predIdx] = vector<double>(nPerm);
    for (unsigned int permIdx = 0; permIdx != nPerm; permIdx++) {
      saePerm[predIdx][permIdx] = testPerm[predIdx][permIdx]->absError;
    }
  }

  return saePerm;
}


vector<vector<vector<double>>> TestCtg::getMispredPermuted(const vector<vector<unique_ptr<TestCtg>>>& testPerm) {
  PredictorT nPred = testPerm.size();
  unsigned int nPerm = testPerm[0].size();
  CtgT nCtg = testPerm[0][0]->misprediction.size();
  vector<vector<vector<double>>> mispredPerm(nPred);

  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    mispredPerm[predIdx] = vector<vector<double>>(nPerm);
    for (unsigned int permIdx = 0; permIdx != nPerm; permIdx++) {
      mispredPerm[predIdx][permIdx] = vector<double>(nCtg);
      for (CtgT ctg = 0; ctg != nCtg; ctg++) {
	mispredPerm[predIdx][permIdx][ctg] = testPerm[predIdx][permIdx]->misprediction[ctg];
      }
    }
  }

  return mispredPerm;
}


vector<vector<double>> TestCtg::getOOBErrorPermuted(const vector<vector<unique_ptr<TestCtg>>>& testPerm) {
  unsigned int nPred = testPerm.size();
  unsigned int nPerm = testPerm[0].size();
  vector<vector<double>> oobPerm(nPred);

  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    oobPerm[predIdx] = vector<double>(nPerm);
    for (unsigned int permIdx = 0; permIdx != nPerm; permIdx++) {
      oobPerm[predIdx][permIdx] = testPerm[predIdx][permIdx]->oobErr;
    }
  }

  return oobPerm;
}


const vector<double>& ForestPredictionReg::getQPred() const {
  return quant->getQPred();
}


const vector<double>& ForestPredictionReg::getQEst() const {
  return quant->getQEst();
}


void ForestPrediction::init(bool indexing) {
  reportIndices = indexing;
}


void ForestPrediction::deInit() {
  reportIndices = false;
}


void CtgProb::init(bool doProb) {
  reportProbabilities = doProb;
}


void CtgProb::deInit() {
  reportProbabilities = false;
}


CtgProb::CtgProb(const Sampler* sampler,
		 size_t nObs,
		 bool reportAuxiliary) :
  nCtg(sampler->getNCtg()),
  probDefault(reinterpret_cast<const ResponseCtg*>(sampler->getResponse())->ctgProb()),
  probs(vector<double>((reportAuxiliary && reportProbabilities) ? nObs * nCtg : 0)) {
}


void CtgProb::predictRow(size_t obsIdx,
			 const vector<double>& numVec,
			 unsigned int nEst) {
  if (isEmpty())
    return;

  double* probRow = &probs[obsIdx * nCtg];
  if (nEst == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / accumulate(numVec.begin(), numVec.end(), 0.0);
    for (PredictorT ctg = 0; ctg != nCtg; ctg++)
      probRow[ctg] = numVec[ctg] * scale;
  }
}


void CtgProb::assignBinary(size_t obsIdx, double p1) {
  double* probRow = &probs[obsIdx * 2];
  probRow[0] = 1.0 - p1;
  probRow[1] = p1;
}


void CtgProb::applyDefault(double probPredict[]) const {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}
