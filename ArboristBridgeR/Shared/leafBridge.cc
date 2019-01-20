// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file leafBridge.cc

   @brief C++ interface to R entry for Leaf methods.

   @author Mark Seligman
 */

#include "leafBridge.h"
#include "leaf.h"
#include "predict.h"
#include "quant.h"

bool LBTrain::thin = false;

LBTrain::LBTrain(unsigned int nTree) :
  nodeHeight(IntegerVector(nTree)),
  nodeRaw(RawVector(0)),
  bagHeight(IntegerVector(nTree)),
  blRaw(RawVector(0)) {
  fill(bagHeight.begin(), bagHeight.end(), 0ul);
}

void LBTrain::init(bool thin_) {
  thin = thin_;
}


void LBTrain::deInit() {
  thin = false;
}

LBTrainReg::LBTrainReg(const NumericVector& yTrain_,
                       unsigned int nTree) :
  LBTrain(nTree),
  yTrain(yTrain_) {
}

LBTrainCtg::LBTrainCtg(const IntegerVector& yTrain_,
                       unsigned int nTree) :
  LBTrain(nTree),
  weight(NumericVector(0)),
  weightSize(0),
  yTrain(yTrain_) {
}

void LBTrain::consume(const LeafTrain* leaf,
                      unsigned int tIdx,
                      double scale) {
  writeScore(leaf, tIdx);
  writeNode(leaf, tIdx, scale);
  writeBagLeaf(leaf, tIdx, scale);
}


void LBTrain::writeScore(const LeafTrain* leaf,
                         unsigned int tIdx) {
  //leaf->cacheScore();
}


void LBTrain::writeNode(const LeafTrain* leaf,
                        unsigned int tIdx,
                        double scale) {
  // Accumulates node heights.
  unsigned int i = tIdx;
  for (auto th : leaf->getLeafHeight()) {
    nodeHeight[i++] = th + (tIdx == 0 ? 0 : nodeHeight[tIdx-1]);
  }

  // Writes leaf nodes as raw.
  size_t nodeOff = tIdx == 0 ? 0 : nodeHeight[tIdx-1] * sizeof(Leaf);
  size_t nodeBytes = leaf->getLeafHeight().back() * sizeof(Leaf);
  if (nodeOff + nodeBytes > static_cast<size_t>(nodeRaw.length())) {
    RawVector temp(scale * (nodeOff + nodeBytes));
    for (size_t i = 0; i < nodeOff; i++)
      temp[i] = nodeRaw[i];
    nodeRaw = move(temp);
  }
  leaf->cacheNodeRaw(&nodeRaw[nodeOff]);
 }


void LBTrain::writeBagLeaf(const LeafTrain* leaf,
                      unsigned int tIdx,
                      double scale) {
  // Thin leaves forgo writing bag state.
  if (thin)
    return;

  auto i = tIdx;
  for (auto bh : leaf->getBagHeight()) {
    bagHeight[i++] = bh + (tIdx == 0 ? 0 : bagHeight[tIdx-1]);
  }
  // Writes BagLeaf records as raw.
  size_t blOff = tIdx == 0 ? 0 : bagHeight[tIdx-1] * sizeof(BagLeaf);
  size_t bagBytes = leaf->getBagHeight().back() * sizeof(BagLeaf);
  if (blOff + bagBytes > static_cast<size_t>(blRaw.length())) {
    RawVector temp(scale * (blOff + bagBytes));
    for (size_t i = 0; i < blOff; i++)
      temp[i] = blRaw[i];
    blRaw = move(temp);
  }
  leaf->cacheBLRaw(&blRaw[blOff]);
}


void LBTrainReg::consume(const LeafTrain* leaf,
                         unsigned int tIdx,
                         double scale) {
  LBTrain::consume(leaf, tIdx, scale);
}

void LBTrainCtg::consume(const LeafTrain* leaf,
                         unsigned int tIdx,
                         double scale) {
  LBTrain::consume(leaf, tIdx, scale);
  writeWeight(static_cast<const LeafTrainCtg*>(leaf), tIdx, scale);
}

void LBTrainCtg::writeWeight(const LeafTrainCtg* leaf,
                             unsigned int tIdx,
                             double scale) {
  auto sizeLoc = static_cast<R_xlen_t>(leaf->getProbSize());
  if (weightSize + sizeLoc > weight.length()) {
    NumericVector temp(scale * (weightSize + sizeLoc));
    for (int i = 0; i < weightSize; i++)
      temp[i] = weight[i];
    weight = move(temp);
  }
  leaf->dumpProb(&weight[weightSize]);
  weightSize += sizeLoc;
}


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
List LBTrainReg::wrap() {
  List leaf =
    List::create(_["nodeHeight"] = move(nodeHeight),
                 _["node"] = move(nodeRaw),
                 _["bagHeight"] = move(bagHeight),
                 _["bagLeaf"] = move(blRaw),
                 _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
List LBTrainCtg::wrap() {
  List leaf =
    List::create(_["nodeHeight"] = move(nodeHeight),
                 _["node"] = move(nodeRaw),
                 _["bagHeight"] = move(bagHeight),
                 _["bagLeaf"] = move(blRaw),
                 _["weight"] = move(weight),
                 _["levels"] = as<CharacterVector>(yTrain.attr("levels"))
                 );
  leaf.attr("class") = "LeafCtg";

  return leaf;
}

/**
   @brief References front-end member arrays and instantiates
   bridge-specific LeafReg handle.
 */
unique_ptr<LeafRegBridge> LeafRegBridge::unwrap(const List &lTrain,
                                                unsigned int nRow) {
  List lLeaf = lTrain["leaf"];
  Legal(lLeaf);
  return make_unique<LeafRegBridge>(IntegerVector((SEXP) lLeaf["nodeHeight"]),
                                    RawVector((SEXP) lLeaf["node"]),
                                    IntegerVector((SEXP) lLeaf["bagHeight"]),
                                    RawVector((SEXP) lLeaf["bagLeaf"]),
                                    NumericVector((SEXP) lLeaf["yTrain"]),
                                    nRow);
}


SEXP LeafRegBridge::Legal(const List &leaf) {
  BEGIN_RCPP

  if (!leaf.inherits("LeafReg")) {
    stop("Expecting LeafReg");
  }

  END_RCPP
}


/**
   @brief Constructor caches front-end vectors and instantiates a Leaf member.
 */
LeafRegBridge::LeafRegBridge(const IntegerVector& feNodeHeight_,
                             const RawVector &feNode_,
                             const IntegerVector& feBagHeight_,
                             const RawVector &feBagLeaf_,
                             const NumericVector &yTrain_,
                             unsigned int rowPredict) :
  LeafBridge(0),
  feNodeHeight(feNodeHeight_),
  feNode(feNode_),
  feBagHeight(feBagHeight_),
  feBagLeaf(feBagLeaf_),
  yTrain(yTrain_) {

  leaf = move(make_unique<LeafFrameReg>((unsigned int *) &feNodeHeight[0],
                                   feNodeHeight.length(),
                                   (Leaf*) &feNode[0],
                                   (unsigned int*) &feBagHeight[0],
                                   (BagLeaf*) &feBagLeaf[0],
                                   &yTrain_[0],
                                   mean(yTrain_),
                                   rowPredict));
}


LeafRegBridge::~LeafRegBridge() {
}

/**
   @brief References front-end vectors and instantiates bridge-specific
   LeafCtg handle.

   @return 
 */
unique_ptr<LeafCtgBridge> LeafCtgBridge::unwrap(const List &sTrain,
                                                unsigned int nRow,
                                                bool doProb) {
  List lLeaf = sTrain["leaf"];
  Legal(lLeaf);
  return make_unique<LeafCtgBridge>(IntegerVector((SEXP) lLeaf["nodeHeight"]),
                                    RawVector((SEXP) lLeaf["node"]),
                                    IntegerVector((SEXP) lLeaf["bagHeight"]),
                                    RawVector((SEXP) lLeaf["bagLeaf"]),
                                    NumericVector((SEXP) lLeaf["weight"]),
                                    CharacterVector((SEXP) lLeaf["levels"]),
                                    nRow,
                                    doProb);
}


/**
   @brief Ensures front end holds a LeafCtg.
 */
SEXP LeafCtgBridge::Legal(const List &leaf) {
  BEGIN_RCPP

  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }

  END_RCPP
}


/**
   @brief Constructor caches front-end vectors and instantiates a Leaf member.
 */
LeafCtgBridge::LeafCtgBridge(const IntegerVector& feNodeHeight_,
                             const RawVector &feNode_,
                             const IntegerVector& feBagHeight_,
                             const RawVector &feBagLeaf_,
                             const NumericVector &feWeight_,
                             const CharacterVector &feLevels_,
                             unsigned int rowPredict,
                             bool doProb) :
  LeafBridge(0),
  feNodeHeight(feNodeHeight_),
  feNode(feNode_),
  feBagHeight(feBagHeight_),
  feBagLeaf(feBagLeaf_),
  feWeight(feWeight_),
  levelsTrain(feLevels_) {
  leaf = move(make_unique<LeafFrameCtg>((unsigned int *) &feNodeHeight[0],
                                   feNodeHeight.length(),
                                   (Leaf*) &feNode[0],
                                   (unsigned int*) &feBagHeight[0],
                                   (BagLeaf*) &feBagLeaf[0],
                                   &feWeight[0],
                                   levelsTrain.length(),
                                   rowPredict,
                                   doProb));
}

LeafCtgBridge::~LeafCtgBridge() {
}

List LeafRegBridge::Summary(SEXP sYTest, const Quant *quant) {
  BEGIN_RCPP
  List prediction;
  if (Rf_isNull(sYTest)) {
    prediction = List::create(
                              _["yPred"] = leaf->YPred(),
                              _["qPred"] = QPred(quant)
                              );
    prediction.attr("class") = "PredictReg";
  }
  else { // Validation/testing
    NumericVector yTest(sYTest);
    double rsq, mae;
    double mse = MSE(leaf->YPred(), yTest, rsq, mae);
    prediction = List::create(
                              _["yPred"] = leaf->YPred(),
                              _["mse"] = mse,
                              _["mae"] = mae,
                              _["rsq"] = rsq,
                              _["qPred"] = QPred(quant)
                              );
    prediction.attr("class") = "ValidReg";
  }

  return prediction;
  END_RCPP
}


/**
   @brief Builds a NumericMatrix representation of the quantile predictions.

   @return transposed core matrix if quantiles requested, else empty matrix.
 */
NumericMatrix LeafRegBridge::QPred(const Quant *quant) {
  BEGIN_RCPP

  return  quant == nullptr ? NumericMatrix(0) : transpose(NumericMatrix(quant->NQuant(), leaf->rowPredict(), quant->QPred()));
  END_RCPP
}

/**
   @brief Utility for computing mean-square error of prediction.
   
   @param yPred is the prediction.

   @param rsq[out] is the r-squared statistic.

   @param mae[out] is the mean absolute error.

   @return mean squared error.
 */
double LeafRegBridge::MSE(const vector<double> &yPred,
                          const NumericVector &yTest,
                          double &rsq,
                          double &mae) {
  double sse = 0.0;
  mae = 0.0;
  unsigned int rowPred = yTest.length();
  for (unsigned int i = 0; i < rowPred; i++) {
    double error = yTest[i] - yPred[i];
    sse += error * error;
    mae += abs(error);
  }
  rsq = 1.0 - sse / (var(yTest) * (rowPred - 1.0));
  mae /= rowPred;

  return sse / rowPred;
}

/**
   @param sYTest is the one-based test vector, possibly null.

   @param rowNames are the row names of the test data.

   @return list of summary entries.   
 */
List LeafCtgBridge::Summary(SEXP sYTest, const List &signature) {
  BEGIN_RCPP

  leaf->vote();
  CharacterVector rowNames = CharacterVector((SEXP) signature["rowNames"]);
  IntegerVector yPredZero(leaf->YPred().begin(), leaf->YPred().end());
  IntegerVector yPredOne = yPredZero + 1;
  yPredOne.attr("class") = "factor";
  yPredOne.attr("levels") = levelsTrain;
  List prediction;
  if (!Rf_isNull(sYTest)) {
    auto testCtg = make_unique<TestCtg>(sYTest, leaf->rowPredict(), getLevelsTrain());
    testCtg->Validate(leaf.get(), leaf->YPred());
    prediction = List::create(
                              _["yPred"] = yPredOne,
                              _["census"] = Census(rowNames),
                              _["prob"] = Prob(rowNames),
                              _["confusion"] = testCtg->Confusion(),
                              _["misprediction"] = testCtg->MisPred(),
                              _["oobError"] = testCtg->OOB(leaf->YPred())
    );
    prediction.attr("class") = "ValidCtg";
  }
  else {
    prediction = List::create(
                      _["yPred"] = yPredOne,
                      _["census"] = Census(rowNames),
                      _["prob"] = Prob(rowNames)
   );
   prediction.attr("class") = "PredictCtg";
  }

  return prediction;
  END_RCPP
}


TestCtg::TestCtg(SEXP sYTest,
                 unsigned int _rowPredict,
                 const CharacterVector &_levelsTrain) :
  rowPredict(_rowPredict),
  levelsTrain(_levelsTrain),
  yTestOne(sYTest),
  levels(CharacterVector((SEXP) yTestOne.attr("levels"))),
  nCtg(levels.length()),
  test2Merged(MergeLevels(levels, levelsTrain)),
  yTestZero(Reconcile(test2Merged, yTestOne)),
  ctgMerged(max(yTestZero) + 1),
  misPred(NumericVector(ctgMerged)),
  confusion(move(vector<unsigned int>(rowPredict * ctgMerged))) {
}


/**
   @brief Fills in confusion matrix and misprediction vector.

   @param yPred contains the zero-based predictions.

   @param yTest contains the zero-based, reconciled test response.

   @param confusion is an uninitialized nRow x ctgMerged matrix

   @param misPred is an unitialized vector of width ctgMerged.

   @return void.
*/
void TestCtg::Validate(LeafFrameCtg *leaf, const vector<unsigned int> &yPred) {
  fill(confusion.begin(), confusion.end(), 0);
  for (unsigned int row = 0; row < rowPredict; row++) {
    confusion[leaf->ctgIdx(yTestZero[row], yPred[row])]++;
  }

  // Fills in misprediction rates for all 'ctgMerged' testing categories.
  // Polls all 'ctgTrain' possible predictions.
  //
  for (unsigned int ctgRec = 0; ctgRec < ctgMerged; ctgRec++) {
    unsigned int numWrong = 0;
    unsigned int numRight = 0;
    for (unsigned int ctgPred = 0; ctgPred < leaf->getCtgTrain(); ctgPred++) {
      if (ctgPred != ctgRec) {  // Misprediction iff off-diagonal.
        numWrong += confusion[leaf->ctgIdx(ctgRec, ctgPred)];
      }
      else {
        numRight = confusion[leaf->ctgIdx(ctgRec, ctgPred)];
      }
    }
    misPred[ctgRec] = numWrong + numRight == 0 ? 0.0 : double(numWrong) / double(numWrong + numRight);
  }
}


/**
   @brief Produces census summary, which is common to all categorical
   prediction.

   @return void.
 */
IntegerMatrix LeafCtgBridge::Census(const CharacterVector &rowNames) {
  IntegerMatrix census = transpose(IntegerMatrix(leaf->getCtgTrain(), leaf->rowPredict(), leaf->Census()));
  census.attr("dimnames") = List::create(rowNames, levelsTrain);
  return census;
}


/**
   @param rowNames decorates the returned matrix.

   @return probability matrix iff requested.
 */
NumericMatrix LeafCtgBridge::Prob(const CharacterVector &rowNames) {
  if (!leaf->Prob().empty()) {
    NumericMatrix prob = transpose(NumericMatrix(leaf->getCtgTrain(), leaf->rowPredict(), &(leaf->Prob())[0]));
    prob.attr("dimnames") = List::create(rowNames, levelsTrain);
    return prob;
  }
  else {
    return NumericMatrix(0);
  }
}


IntegerVector TestCtg::MergeLevels(const CharacterVector &levelsTest,
                                   const CharacterVector &levelsTrain) {
  BEGIN_RCPP
  IntegerVector test2Merged(match(levelsTest, levelsTrain));
  IntegerVector sq = seq(0, test2Merged.length() - 1);
  IntegerVector idxNA = sq[is_na(test2Merged)];
  if (idxNA.length() > 0) {
    warning("Uninferable test levels not encountered in training");
    int proxy = levelsTrain.length() + 1;
    for (R_len_t i = 0; i < idxNA.length(); i++) {
      int idx = idxNA[i];
      test2Merged[idx] = proxy++;
    }
  }
  return test2Merged - 1;
  END_RCPP
}

/**
   @brief Determines summary array dimensions by reconciling cardinalities
   of training and test reponses.

   @return reconciled test vector.
 */
IntegerVector TestCtg::Reconcile(const IntegerVector &test2Merged,
                                 const IntegerVector &yTestOne) {
  BEGIN_RCPP
  IntegerVector yZero = yTestOne -1;
  IntegerVector yZeroOut(yZero.length());
  for (unsigned int i = 0; i < yZero.length(); i++) {
    yZeroOut[i] = test2Merged[yZero[i]];
  }
  return yZeroOut;
  END_RCPP
}


/**
   @brief Produces summary information specific to testing:  mispredction
   vector and confusion matrix.

   @return void.
 */
IntegerMatrix TestCtg::Confusion() {
  BEGIN_RCPP
  unsigned int ctgTrain = levelsTrain.length();
  IntegerMatrix conf = transpose(IntegerMatrix(ctgTrain, nCtg, &confusion[0]));
  IntegerMatrix confOut(nCtg, ctgTrain);
  for (unsigned int i = 0; i < nCtg; i++) {
    confOut(i, _) = conf(test2Merged[i], _);
  }
  confOut.attr("dimnames") = List::create(levels, levelsTrain);

  return confOut;
  END_RCPP
}


NumericVector TestCtg::MisPred() {
  BEGIN_RCPP
  NumericVector misPredOut = misPred[test2Merged];
  misPredOut.attr("names") = levels;
  return misPredOut;
  END_RCPP
}


/**
   @brief Computes the mean number of mispredictions.

   @param yPred is the zero-based prediction vector derived by the core.

   @return OOB as mean number of mispredictions, if testing, otherwise 0.0.
 */
double TestCtg::OOB(const vector<unsigned int> &yPred) const {
  unsigned int missed = 0;
  for (unsigned int i = 0; i < rowPredict; i++) {
    missed += (unsigned int) yTestZero[i] != yPred[i];
  }

  return double(missed) / rowPredict;  // Caller precludes zero length.
}


unique_ptr<LeafCtgBridge> LeafCtgBridge::unwrap(const List &lTrain,
                                                const BitMatrix *baggedRows) {
  List lLeaf((SEXP) lTrain["leaf"]);
  Legal(lLeaf);
  return make_unique<LeafCtgBridge>(IntegerVector((SEXP) lLeaf["nodeHeight"]),
                                    RawVector((SEXP) lLeaf["node"]),
                                    IntegerVector((SEXP) lLeaf["bagHeight"]),
                                    RawVector((SEXP) lLeaf["bagLeaf"]),
                                    NumericVector((SEXP) lLeaf["weight"]),
                                    CharacterVector((SEXP) lLeaf["levels"]),
                                    baggedRows);
}
 

/**
   @brief Constructor caches front-end vectors and instantiates a Leaf member.
 */
LeafCtgBridge::LeafCtgBridge(const IntegerVector& feNodeHeight_,
                             const RawVector& feNode_,
                             const IntegerVector& feBagHeight_,
                             const RawVector& feBagLeaf_,
                             const NumericVector& feWeight_,
                             const CharacterVector& feLevels_,
                             const BitMatrix* baggedRows) :
  LeafBridge(feNodeHeight_.length()),
  feNodeHeight(feNodeHeight_),
  feNode(feNode_),
  feBagHeight(feBagHeight_),
  feBagLeaf(feBagLeaf_),
  feWeight(feWeight_),
  levelsTrain(feLevels_),
  scoreTree(vector<vector<double > >(feNodeHeight.length())),
  weightTree(vector<vector<double> >(feNodeHeight.length())) {
  leaf = move(make_unique<LeafFrameCtg>((unsigned int*) &feNodeHeight[0],
                                   feNodeHeight.length(),
                                   (Leaf*) &feNode[0],
                                   (unsigned int*) &feBagHeight[0],
                                   (BagLeaf*) &feBagLeaf[0],
                                   &feWeight[0],
                                   levelsTrain.length(),
                                   0,
                                   false));
  leaf->dump(baggedRows, rowTree, sCountTree, scoreTree, extentTree, weightTree);
}


unique_ptr<LeafRegBridge> LeafRegBridge::unwrap(const List &lTrain,
                                                const BitMatrix *baggedRows) {
  List lLeaf((SEXP) lTrain["leaf"]);
  Legal(lLeaf);
  
  return make_unique<LeafRegBridge>(IntegerVector((SEXP) lLeaf["nodeHeight"]),
                                    RawVector((SEXP) lLeaf["node"]),
                                    IntegerVector((SEXP) lLeaf["bagHeight"]),
                                    RawVector((SEXP) lLeaf["bagLeaf"]),
                                    NumericVector((SEXP) lLeaf["yTrain"]),
                                    baggedRows);
}
 

/**
   @brief Constructor instantiates leaves for export only:
   no prediction.
 */
LeafRegBridge::LeafRegBridge(const IntegerVector& feNodeHeight_,
                             const RawVector &feNode_,
                             const IntegerVector& feBagHeight_,
                             const RawVector &feBagLeaf_,
                             const NumericVector &yTrain_,
                             const BitMatrix *baggedRows) :
  LeafBridge(feNodeHeight_.length()),
  feNodeHeight(feNodeHeight_),
  feNode(feNode_),
  feBagHeight(feBagHeight_),
  feBagLeaf(feBagLeaf_),
  yTrain(yTrain_),
  scoreTree(vector<vector<double > >(feNodeHeight.length())) {
  leaf = move(make_unique<LeafFrameReg>((unsigned int *) &feNodeHeight[0],
                                   feNodeHeight.length(),
                                   (Leaf*) &feNode[0],
                                   (unsigned int*) &feBagHeight[0],
                                   (BagLeaf*) &feBagLeaf[0],
                                   &yTrain[0],
                                   mean(yTrain),
                                   0));
  leaf->dump(baggedRows, rowTree, sCountTree, scoreTree, extentTree);
}


LeafBridge::LeafBridge(unsigned int exportLength) :
  rowTree(vector<vector<unsigned int> >(exportLength)),
  sCountTree(vector<vector<unsigned int> >(exportLength)),
  extentTree(vector<vector<unsigned int> >(exportLength)) {
}
