// Copyright (C)  2012-2020   Mark Seligman
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
   @file leafRf.cc

   @brief C++ interface to R entry for Leaf methods.

   @author Mark Seligman
 */

#include "leaftrainRf.h"
#include "trainbridge.h"


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


void LBTrain::consume(const TrainChunk* train,
                      unsigned int tIdx,
                      double scale) {
  writeNode(train, tIdx, scale);
  writeBagSample(train, tIdx, scale);
}


void LBTrain::writeNode(const TrainChunk* train,
                        unsigned int tIdx,
                        double scale) {
  // Accumulates node heights.
  train->writeHeight((unsigned int*) &nodeHeight[0], tIdx);

  // Reallocates forest-wide buffer if estimated size insufficient.
  size_t nodeOff, nodeBytes;
  if (!train->leafFits((unsigned int*) &nodeHeight[0], tIdx, static_cast<size_t>(nodeRaw.length()), nodeOff, nodeBytes)) {
    nodeRaw = move(rawResize(&nodeRaw[0], nodeOff, nodeBytes, scale));
  }

  // Writes leaves as raw.
  train->dumpLeafRaw(&nodeRaw[nodeOff]);
}


RawVector LBTrain::rawResize(const unsigned char* raw, size_t offset, size_t bytes, double scale) {
  RawVector temp(scale * (offset + bytes));
  for (size_t i = 0; i < offset; i++)
    temp[i] = raw[i];

  return temp;
}


void LBTrain::writeBagSample(const TrainChunk* train,
                             unsigned int tIdx,
                             double scale) {
  // Thin leaves forgo writing bag state.
  if (thin)
    return;

  train->writeBagHeight((unsigned int*) &bagHeight[0], tIdx);

  // Writes BagSample records as raw.
  size_t blOff, bagBytes;
  if (!train->bagSampleFits((unsigned int*) &bagHeight[0], tIdx, static_cast<size_t>(blRaw.length()), blOff, bagBytes)) {
    blRaw = move(rawResize(&blRaw[0], blOff, bagBytes, scale));
  }
  train->dumpBagLeafRaw(&blRaw[blOff]);
}


void LBTrainReg::consume(const TrainChunk* train,
                         unsigned int tIdx,
                         double scale) {
  LBTrain::consume(train, tIdx, scale);
}


void LBTrainCtg::consume(const TrainChunk* train,
                         unsigned int tIdx,
                         double scale) {
  LBTrain::consume(train, tIdx, scale);
  writeWeight(train, scale);
}


void LBTrainCtg::writeWeight(const TrainChunk* train,
                             double scale) {
  auto sizeLoc = train->getWeightSize();
  if (weightSize + sizeLoc > static_cast<size_t>(weight.length())) {
    weight = move(numericResize(&weight[0], weightSize, sizeLoc, scale));
  }
  train->dumpLeafWeight(&weight[weightSize]);
  weightSize += sizeLoc;
}


NumericVector LBTrainCtg::numericResize(const double* num, size_t offset, size_t elts, double scale) {
  NumericVector temp(scale * (offset + elts));
  for (size_t i = 0; i < offset; i++) {
    temp[i] = num[i];
  }
  return temp;
}


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
List LBTrainReg::wrap() {
  BEGIN_RCPP
  List leaf =
    List::create(_["nodeHeight"] = move(nodeHeight),
                 _["node"] = move(nodeRaw),
                 _["bagHeight"] = move(bagHeight),
                 _["bagSample"] = move(blRaw),
                 _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
  END_RCPP
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
List LBTrainCtg::wrap() {
  BEGIN_RCPP
  List leaf =
    List::create(_["nodeHeight"] = move(nodeHeight),
                 _["node"] = move(nodeRaw),
                 _["bagHeight"] = move(bagHeight),
                 _["bagSample"] = move(blRaw),
                 _["weight"] = move(weight),
                 _["levels"] = as<CharacterVector>(yTrain.attr("levels"))
                 );
  leaf.attr("class") = "LeafCtg";

  return leaf;
  END_RCPP
}
