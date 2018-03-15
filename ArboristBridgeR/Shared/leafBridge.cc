// Copyright (C)  2012-2018   Mark Seligman
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


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
List LeafBridge::Wrap(LeafTrainReg *leafReg,
		      const NumericVector &yTrain) {
  RawVector leafRaw(leafReg->NodeBytes());
  RawVector blRaw(leafReg->BLBytes());
  RawVector bbRaw(leafReg->BagBytes());
  leafReg->Serialize((unsigned char *) &leafRaw[0], (unsigned char*) &blRaw[0], (unsigned char *) &bbRaw[0]);
  List leaf = List::create(
   _["origin"] = leafReg->Origin(),
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
List LeafBridge::Wrap(LeafTrainCtg *leafCtg,
		      const CharacterVector &levels) {
  RawVector leafRaw(leafCtg->NodeBytes());
  RawVector blRaw(leafCtg->BLBytes());
  RawVector bbRaw(leafCtg->BagBytes());
  leafCtg->Serialize((unsigned char *) &leafRaw[0], (unsigned char *) &blRaw[0], (unsigned char *) &bbRaw[0]);
  List leaf = List::create(
   _["origin"] = leafCtg->Origin(),
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["weight"] = leafCtg->Weight(),
   _["rowTrain"] = leafCtg->RowTrain(),
   _["levels"] = levels
   );
  leaf.attr("class") = "LeafCtg";

  return leaf;
}


List LeafRegBridge::Prediction(const List &list,
			       SEXP sYTest,
			       Predict *predict) {
  auto leafReg = Unwrap(list, predict->NRow());
  predict->PredictAcross(leafReg->GetLeaf());
  return move(leafReg->Summary(sYTest));
}

List LeafRegBridge::Prediction(const List &list,
			       SEXP sYTest,
			       Predict *predict,
			       const NumericVector &quantVec,
			       unsigned int qBin) {
  auto leafReg = LeafRegBridge::Unwrap(list,
				       predict->NRow(),
				       quantVec,
				       qBin);
  predict->PredictAcross(leafReg->GetLeaf());
  return move(leafReg->Summary(sYTest));
}


/**
   @brief References front-end member arrays and instantiates
   bridge-specific LeafReg handle.
 */
unique_ptr<LeafRegBridge> LeafRegBridge::Unwrap(const List &leaf,
						unsigned int nRow) {
  Legal(leaf);
  return make_unique<LeafRegBridge>(IntegerVector((SEXP) leaf["origin"]),
				    RawVector((SEXP) leaf["bagBits"]),
				    RawVector((SEXP) leaf["bagLeaf"]),
				    RawVector((SEXP) leaf["node"]),
				    NumericVector((SEXP) leaf["yTrain"]),
				    nRow);
}


/**
   @brief References front-end member arrays and instantiates
   bridge-specific LeafReg handle.
 */
unique_ptr<LeafRegBridge> LeafRegBridge::Unwrap(const List &leaf,
						unsigned int nRow,
						const NumericVector &sQuantVec,
						unsigned int qBin) {
  Legal(leaf);
  return make_unique<LeafRegBridge>(IntegerVector((SEXP) leaf["origin"]),
				    RawVector((SEXP) leaf["bagBits"]),
				    RawVector((SEXP) leaf["bagLeaf"]),
				    RawVector((SEXP) leaf["node"]),
				    NumericVector((SEXP) leaf["yTrain"]),
				    nRow,
				    NumericVector(sQuantVec),
				    qBin);
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
LeafRegBridge::LeafRegBridge(const IntegerVector &_feOrig,
			     const RawVector &_feBagBits,
			     const RawVector &_feBagLeaf,
			     const RawVector &_feNode,
			     const NumericVector &_yTrain,
			     unsigned int _rowPredict) :
  feOrig(_feOrig),
  feBagBits(_feBagBits),
  feBagLeaf(_feBagLeaf),
  feNode(_feNode),
  yTrain(_yTrain) {

  leaf = move(make_unique<LeafReg>((unsigned int *) &feOrig[0],
				   feOrig.length(),
				   (LeafNode*) &feNode[0],
				   feNode.length()/sizeof(LeafNode),
				   (BagLeaf*) &feBagLeaf[0],
				   feBagLeaf.length() / sizeof(BagLeaf),
				   (unsigned int *) &feBagBits[0],
				   &_yTrain[0],
				   _yTrain.length(),
				   mean(_yTrain),
				   _rowPredict));
}


/**
   @brief Constructor caches front-end vectors and instantiates a Leaf member.
 */
LeafRegBridge::LeafRegBridge(const IntegerVector &_feOrig,
			     const RawVector &_feBagBits,
			     const RawVector &_feBagLeaf,
			     const RawVector &_feNode,
			     const NumericVector &_yTrain,
			     unsigned int _rowPredict,
			     const NumericVector &quantiles,
			     const unsigned int qBin) :
  feOrig(_feOrig),
  feBagBits(_feBagBits),
  feBagLeaf(_feBagLeaf),
  feNode(_feNode),
  yTrain(_yTrain) {
  leaf = move(make_unique<LeafReg>((unsigned int *) &feOrig[0],
				   feOrig.length(),
				   (LeafNode*) &feNode[0],
				   feNode.length()/sizeof(LeafNode),
				   (BagLeaf*) &feBagLeaf[0],
				   feBagLeaf.length() / sizeof(BagLeaf),
				   (unsigned int *) &feBagBits[0],
				   &_yTrain[0],
				   _yTrain.length(),
				   mean(_yTrain),
				   _rowPredict,
				   as<vector<double> >(quantiles),
				   qBin));
}


List LeafCtgBridge::Prediction(const List &list,
			       SEXP sYTest,
			       const List &signature,
			       Predict *predict,
			       bool doProb) {
  auto leafCtg = LeafCtgBridge::Unwrap(list, predict->NRow(), doProb);
  predict->PredictAcross(leafCtg->GetLeaf());
  return move(leafCtg->Summary(sYTest, signature));
}


/**
   @brief References front-end vectors and instantiates bridge-specific
   LeafCtg handle.

   @return 
 */
unique_ptr<LeafCtgBridge> LeafCtgBridge::Unwrap(const List &leaf,
						unsigned int nRow,
						bool doProb) {
  Legal(leaf);
  return make_unique<LeafCtgBridge>(
			  IntegerVector((SEXP) leaf["origin"]),
			   RawVector((SEXP) leaf["bagBits"]),
			   RawVector((SEXP) leaf["bagLeaf"]),
			   RawVector((SEXP) leaf["node"]),
			   NumericVector((SEXP) leaf["weight"]),
			   as<unsigned int>((SEXP) leaf["rowTrain"]),
			   CharacterVector((SEXP) leaf["levels"]),
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
LeafCtgBridge::LeafCtgBridge(const IntegerVector &_feOrig,
			     const RawVector &_feBagBits,
			     const RawVector &_feBagLeaf,
			     const RawVector &_feNode,
			     const NumericVector &_feWeight,
			     unsigned int _feRowTrain,
			     const CharacterVector &_feLevels,
			     unsigned int _rowPredict,
			     bool doProb) :
  feOrig(_feOrig),
  feBagBits(_feBagBits),
  feBagLeaf(_feBagLeaf),
  feNode(_feNode),
  feWeight(_feWeight),
  levelsTrain(_feLevels) {
  leaf = move(make_unique<LeafCtg>((unsigned int *) &feOrig[0],
				 feOrig.length(),
				 (LeafNode*) &feNode[0],
				 feNode.length()/sizeof(LeafNode),
				 (BagLeaf*) &feBagLeaf[0],
				 feBagLeaf.length() / sizeof(BagLeaf),
				   (unsigned int *) &feBagBits[0],
				 _feRowTrain,
				 &feWeight[0],
				 levelsTrain.length(),
				   _rowPredict,
				   doProb));
}


List LeafRegBridge::Summary(SEXP sYTest) {
  BEGIN_RCPP
  List prediction;
  if (Rf_isNull(sYTest)) {
    prediction = List::create(
			      _["yPred"] = leaf->YPred(),
			      _["qPred"] = QPred()
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
			      _["qPred"] = QPred()
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
NumericMatrix LeafRegBridge::QPred() {
  BEGIN_RCPP
    unsigned int nQuantile;
  const auto qPred = leaf->GetQuant(nQuantile);
  return  nQuantile == 0 ? NumericMatrix(0) : transpose(NumericMatrix(nQuantile, leaf->RowPredict(), qPred));
  END_RCPP
}

/**
   @brief Utility for computing mean-square error of prediction.
   
   @param yPred is the prediction.

   @param rsq outputs the r-squared statistic.

   @return mean squared error, with output parameter.
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
    leaf->Vote();
    CharacterVector rowNames = CharacterVector((SEXP) signature["rowNames"]);
  IntegerVector yPredZero(leaf->YPred().begin(), leaf->YPred().end());
  IntegerVector yPredOne = yPredZero + 1;
  List prediction;
  if (!Rf_isNull(sYTest)) {
    auto testCtg = make_unique<TestCtg>(sYTest, leaf->RowPredict(), LevelsTrain());
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
  confusion( move(vector<unsigned int>(rowPredict * ctgMerged))) {
}


/**
   @brief Fills in confusion matrix and misprediction vector.

   @param yPred contains the zero-based predictions.

   @param yTest contains the zero-based, reconciled test response.

   @param confusion is an uninitialized nRow x ctgMerged matrix

   @param misPred is an unitialized vector of width ctgMerged.

   @return void.
*/
void TestCtg::Validate(LeafCtg *leaf, const vector<unsigned int> &yPred) {
  fill(confusion.begin(), confusion.end(), 0);
  for (unsigned int row = 0; row < rowPredict; row++) {
    confusion[leaf->TrainIdx(yTestZero[row], yPred[row])]++;
  }

  // Fills in misprediction rates for all 'ctgMerged' testing categories.
  // Polls all 'ctgTrain' possible predictions.
  //
  for (unsigned int ctgRec = 0; ctgRec < ctgMerged; ctgRec++) {
    unsigned int numWrong = 0;
    unsigned int numRight = 0;
    for (unsigned int ctgPred = 0; ctgPred < leaf->CtgTrain(); ctgPred++) {
      if (ctgPred != ctgRec) {  // Misprediction iff off-diagonal.
        numWrong += confusion[leaf->TrainIdx(ctgRec, ctgPred)];
      }
      else {
	numRight = confusion[leaf->TrainIdx(ctgRec, ctgPred)];
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
  IntegerMatrix census = transpose(IntegerMatrix(leaf->CtgTrain(), leaf->RowPredict(), leaf->Census()));
  census.attr("dimnames") = List::create(rowNames, levelsTrain);
  return census;
}


/**
   @param rowNames decorates the returned matrix.

   @return probability matrix iff requested.
 */
NumericMatrix LeafCtgBridge::Prob(const CharacterVector &rowNames) {
  if (!leaf->Prob().empty()) {
    NumericMatrix prob = transpose(NumericMatrix(leaf->CtgTrain(), leaf->RowPredict(), &(leaf->Prob())[0]));
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

   @param yTest is a zero-based NumericVector cached by the bridge.

   @return OOB as mean number of mispredictions, if testing, otherwise 0.0.
 */
double TestCtg::OOB(const vector<unsigned int> &yPred) const {
  unsigned int missed = 0;
  for (unsigned int i = 0; i < rowPredict; i++) {
    missed += (unsigned int) yTestZero[i] != yPred[i];
  }

  return double(missed) / rowPredict;  // Caller precludes zero length.
}


LeafExportCtg::LeafExportCtg(const List &_leaf) :
  LeafCtgBridge(IntegerVector((SEXP) _leaf["origin"]),
		RawVector((SEXP) _leaf["bagBits"]),
		RawVector((SEXP) _leaf["bagLeaf"]),
		RawVector((SEXP) _leaf["node"]),
		NumericVector((SEXP) _leaf["weight"]),
		as<unsigned int>((SEXP) _leaf["rowTrain"]),
		CharacterVector((SEXP) _leaf["levels"]),
		0,
		false),
  nTree(leaf->NTree()),
  rowTree(vector<vector<unsigned int> >(nTree)),
  sCountTree(vector<vector<unsigned int> >(nTree)),
  extentTree(vector<vector<unsigned int> >(nTree)),
  scoreTree(vector<vector<double > >(nTree)),
  weightTree(vector<vector<double> >(nTree)) {
  leaf->Export(rowTrain, rowTree, sCountTree, scoreTree, extentTree, weightTree);
}


LeafExportReg::LeafExportReg(const List &_leaf) :
  LeafRegBridge((SEXP) _leaf["origin"],
		RawVector((SEXP) _leaf["bagBits"]),
		RawVector((SEXP) _leaf["bagLeaf"]),
		RawVector((SEXP) _leaf["node"]),
		NumericVector((SEXP) _leaf["yTrain"]),
		0),
  nTree(leaf->NTree()),
  rowTree(vector<vector<unsigned int> >(nTree)),
  sCountTree(vector<vector<unsigned int> >(nTree)),
  extentTree(vector<vector<unsigned int> >(nTree)),
  scoreTree(vector<vector<double > >( nTree)) {
  leaf->Export(rowTrain, rowTree, sCountTree, scoreTree, extentTree);
}
