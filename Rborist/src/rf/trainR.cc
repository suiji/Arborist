// Copyright (C)  2012-2022   Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file trainR.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "forestbridge.h"
#include "samplerbridge.h"
#include "leafbridge.h"
#include "trainR.h"
#include "trainbridge.h"
#include "samplerR.h"
#include "leafR.h"
#include "forestR.h"
#include "rleframeR.h"
#include "rleframe.h"

bool TrainRf::verbose = false;

RcppExport SEXP rfTrain(const SEXP sDeframe, const SEXP sSampler, const SEXP sArgList) {
  BEGIN_RCPP

  return TrainRf::train(List(sDeframe), List(sSampler), List(sArgList));

  END_RCPP
}


List TrainRf::train(const List& lDeframe, const List& lSampler, const List& argList) {
  BEGIN_RCPP

  return train(argList, SamplerR::unwrapTrain(lSampler, argList), RLEFrameR::unwrap(lDeframe).get());

  END_RCPP
}


List TrainRf::train(const List& argList,
		    unique_ptr<SamplerBridge> sb,
		    const RLEFrame* rleFrame) {
  BEGIN_RCPP

  if (verbose) {
    Rcout << "Beginning training" << endl;
  }
  vector<string> diag;
  unique_ptr<TrainBridge> trainBridge(make_unique<TrainBridge>(rleFrame, as<double>(argList["autoCompress"]), as<bool>(argList["enableCoproc"]), diag));
  initFromArgs(argList, trainBridge.get());

  TrainRf trainRf(sb.get());
  trainRf.trainChunks(sb.get(), trainBridge.get(), as<bool>(argList["thinLeaves"]));
  List outList = trainRf.summarize(trainBridge.get(), diag);

  if (verbose) {
    Rcout << "Training completed" << endl;
  }

  deInit(trainBridge.get());
  return outList;

  END_RCPP
}


// Employs Rcpp-style temporaries for ease of indexing through
// the predMap[] vector.
SEXP TrainRf::initFromArgs(const List& argList,
			   TrainBridge* trainBridge) {
  BEGIN_RCPP

  vector<unsigned int> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  verbose = as<bool>(argList["verbose"]);
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge->initProb(as<unsigned int>(argList["predFixed"]), predProb);

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge->initSplit(as<unsigned int>(argList["minNode"]),
			 as<unsigned int>(argList["nLevel"]),
			 as<double>(argList["minInfo"]),
			 splitQuant);

  trainBridge->initTree(as<unsigned int>(argList["maxLeaf"]));
  trainBridge->initBlock(as<unsigned int>(argList["treeBlock"]));
  trainBridge->initOmp(as<unsigned int>(argList["nThread"]));
  
  if (!Rf_isFactor((SEXP) argList["y"])) {
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    trainBridge->initMono(regMono);
  }

  END_RCPP
}


SEXP TrainRf::deInit(TrainBridge* trainBridge) {
  BEGIN_RCPP

  verbose = false;
  trainBridge->deInit();
  END_RCPP
}


void TrainRf::consume(const ForestBridge& fb,
		      const LeafBridge* lb,
                      unsigned int treeOff,
                      unsigned int chunkSize) const {
  double scale = safeScale(treeOff + chunkSize);
  forest->bridgeConsume(fb, treeOff, scale);
  leaf->bridgeConsume(lb, scale);
  
  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}


void TrainRf::consumeInfo(const TrainedChunk* train) {
  NumericVector infoChunk(train->getPredInfo().begin(), train->getPredInfo().end());
  if (predInfo.length() == 0) {
    predInfo = infoChunk;
  }
  else {
    predInfo = predInfo + infoChunk;
  }
}


List TrainRf::summarize(const TrainBridge* trainBridge,
			const vector<string>& diag) const {
  BEGIN_RCPP
  return List::create(
                      _["predInfo"] = scaleInfo(trainBridge),
                      _["diag"] = diag,
                      _["forest"] = std::move(forest->wrap()),
		      _["predMap"] = std::move(trainBridge->getPredMap()),
		      _["leaf"] = std::move(leaf->wrap())
                      );
  END_RCPP
}


NumericVector TrainRf::scaleInfo(const TrainBridge* trainBridge) const {
  BEGIN_RCPP

  vector<unsigned int> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  // Mapbs back from core order and scales info per-tree.
  return as<NumericVector>(predInfo[predMap]) / nTree;

  END_RCPP
}


TrainRf::TrainRf(const SamplerBridge* sb) :
  nTree(sb->getNTree()),
  leaf(make_unique<LeafR>()),
  forest(make_unique<FBTrain>(sb->getNTree())) {
}


void TrainRf::trainChunks(const SamplerBridge* sb,
			  const TrainBridge* trainBridge,
			  bool thinLeaves) {
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    ForestBridge fb(chunkThis);
    unique_ptr<LeafBridge> lb = LeafBridge::FactoryTrain(sb, thinLeaves);
    auto trainedChunk = trainBridge->train(fb, sb, treeOff, chunkThis, lb.get());
    consume(fb, lb.get(), treeOff, chunkThis);
    consumeInfo(trainedChunk.get());
  }
}
