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
   @file trainBridge.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "trainBridge.h"
#include "framemapBridge.h"
#include "rankedsetBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "coproc.h"

RcppExport SEXP Train(const SEXP sArgList) {
  BEGIN_RCPP

  List argList(sArgList);
  List predBlock(as<List>(argList["predBlock"]));
  List signature(as<List>(predBlock["signature"]));

  // Temporary copy for subscripted access by IntegerVector.
  IntegerVector predMap((SEXP) signature["predMap"]);
  vector<unsigned int> facCard(as<vector<unsigned int> >(predBlock["facCard"]));

  return TrainBridge::Train(argList, predMap, facCard, as<unsigned int>(predBlock["nRow"]));
  END_RCPP
}


List TrainBridge::Train(const List &argList,
			const IntegerVector &predMap,
			const vector<unsigned int> &facCard,
			unsigned int nRow) {
BEGIN_RCPP

  auto frameTrain = FramemapBridge::FactoryTrain(facCard, predMap.length(), nRow);
  vector<string> diag;
  auto coproc = Coproc::Factory(as<bool>(argList["enableCoproc"]), diag);

  auto rankedSet = RankedSetBridge::Unwrap(argList["rankedSet"],
					   as<double>(argList["autoCompress"]),
					   coproc.get(),
					   frameTrain.get());
  Init(argList, predMap);
  List outList;
  if (as<unsigned int>(argList["nCtg"]) > 0) {
    outList =  Classification(argList,
			      frameTrain.get(),
			      rankedSet->GetPair(),
			      predMap,
			      diag);
  }
  else {
    outList =  Regression(argList,
			  frameTrain.get(),
			  rankedSet->GetPair(),
			  predMap,
			  diag);
  }
  Train::DeInit();

  return outList;
END_RCPP
}


// Employs Rcpp-style temporaries for ease of indexing through
// the predMap[] vector.
SEXP TrainBridge::Init(const List &argList, const IntegerVector &predMap) {
  BEGIN_RCPP

  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  Train::InitProb(as<unsigned int>(argList["predFixed"]), predProb);
  
  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  Train::InitCDF(splitQuant);

  vector<double> rowWeight(as<vector<double> >(argList["rowWeight"]));
  Train::InitSample(as<unsigned int>(argList["nSamp"]),
		    rowWeight,
		    as<bool>(argList["withRepl"]));
  Train::InitSplit(as<unsigned int>(argList["minNode"]),
		   as<unsigned int>(argList["nLevel"]),
		   as<double>(argList["minInfo"]));
  Train::InitTree(as<unsigned int>(argList["nSamp"]),
		  as<unsigned int>(argList["minNode"]),
		  as<unsigned int>(argList["maxLeaf"]));
  Train::InitLeaf(as<bool>(argList["thinLeaves"]));
  Train::InitBlock(as<unsigned int>(argList["treeBlock"]));

  unsigned int nCtg = as<unsigned int>(argList["nCtg"]);
  Train::InitCtgWidth(nCtg);
  if (nCtg == 0) { // Regression.
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    Train::InitMono(regMono);
  }

  END_RCPP
}


// Class weighting constructs a proxy response from category frequency.
// The response is then jittered to diminish the possibility of ties
// during scoring.  The magnitude of the jitter, then, should be scaled
// so that no combination of samples can "vote" themselves into a
// false plurality.
//
NumericVector TrainBridge::CtgProxy(const IntegerVector &y,
				    const NumericVector &classWeight) {
  BEGIN_RCPP
    
  NumericVector scaledWeight = clone(classWeight);
  if (is_true(all(classWeight == 0.0))) { // Place-holder for balancing.
    NumericVector tb(table(y));
    for (R_len_t i = 0; i < classWeight.length(); i++) {
      scaledWeight[i] = tb[i] == 0.0 ? 0.0 : 1.0 / tb[i];
    }
  }
  scaledWeight = scaledWeight / sum(scaledWeight);

  unsigned int nRow = y.length();
  NumericVector yWeighted = scaledWeight[y];
  RNGScope scope;
  NumericVector rn(runif(nRow));
  NumericVector proxy = yWeighted + (rn - 0.5) / (2 * nRow * nRow);

  return proxy;
  END_RCPP
}


List TrainBridge::Classification(const List &argList,
				 const FrameTrain *frameTrain,
				 const RankedSet *rankedPair,
				 const IntegerVector &predMap,
				 vector<string> &diag) {
  BEGIN_RCPP
  auto nTree = as<unsigned int>(argList["nTree"]);

  IntegerVector y = IntegerVector((SEXP) argList["y"]);
  NumericVector classWeight = NumericVector((SEXP) argList["classWeight"]);
  IntegerVector yZero = y - 1; // Zero-based translation.
  auto proxy = CtgProxy(yZero, classWeight);
  auto trainCtg = Train::Classification(frameTrain,
					rankedPair,
					&(as<vector<unsigned int> >(yZero))[0],
					&proxy[0],
					classWeight.size(),
					nTree);
  return Summarize(trainCtg.get(), predMap, nTree, y, diag);
  END_RCPP
}


List TrainBridge::Summarize(const TrainCtg *trainCtg,
			    const IntegerVector &predMap,
			    unsigned int nTree,
			    const IntegerVector &y,
			    const vector<string> &diag) {
  BEGIN_RCPP
  return List::create(
      _["predInfo"] = PredInfo(trainCtg->PredInfo(), predMap, nTree),
      _["diag"] = diag,
      _["forest"] = move(ForestBridge::Wrap(trainCtg->Forest())),
      _["leaf"] = move(LeafBridge::Wrap(trainCtg->SubLeaf(),
					as<CharacterVector>(y.attr("levels"))))
		      );

  END_RCPP
}


NumericVector TrainBridge::PredInfo(const vector<double> &predInfo,
					  const IntegerVector &predMap,
					  unsigned int nTree) {
  BEGIN_RCPP
  NumericVector infoOut(predInfo.begin(), predInfo.end());
  infoOut = infoOut / nTree; // Scales info per-tree.
  return infoOut[predMap]; // Maps back from core order.
  END_RCPP
}


List TrainBridge::Regression(const List &argList,
			     const FrameTrain *frameTrain,
			     const RankedSet *rankedPair,
			     const IntegerVector &predMap,
			     vector<string> &diag) {
  BEGIN_RCPP
  auto nTree = as<unsigned int>(argList["nTree"]);
  NumericVector y = NumericVector((SEXP) argList["y"]);
  NumericVector yOrdered = clone(y).sort();
  IntegerVector row2Rank = match(y, yOrdered) - 1;

  auto trainReg = Train::Regression(frameTrain,
				    rankedPair,
				    &y[0],
				    &(as<vector<unsigned int> >(row2Rank))[0],
				    nTree);

  return Summarize(trainReg.get(), predMap, nTree, y, diag);
  END_RCPP
}


List TrainBridge::Summarize(const TrainReg *trainReg,
			    const IntegerVector &predMap,
			    unsigned int nTree,
			    const NumericVector &y,
			    const vector<string> &diag) {
  BEGIN_RCPP
  return List::create(
      _["predInfo"] = PredInfo(trainReg->PredInfo(), predMap, nTree),
      _["diag"] = diag,
      _["forest"] = move(ForestBridge::Wrap(trainReg->Forest())),
      _["leaf"] = move(LeafBridge::Wrap(trainReg->SubLeaf(), y))
  );
  END_RCPP
}
