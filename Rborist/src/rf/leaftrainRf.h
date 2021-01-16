// Copyright (C)  2012-2021  Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file leafRf.h

   @brief C++ class definitions for managing Leaf object.

   @author Mark Seligman

 */

#ifndef RF_LEAF_RF_H
#define RF_LEAF_RF_H

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>
#include <memory>
using namespace std;

/**
   @brief Maintains R-style vectors represting the crescent leaf component
   of the forest during training.
 */
struct LBTrain {
private:
  static bool thin; // User option:  whether to annotate bag state.

  /**
     @brief Consumes core Node recrods and writes as raw data.

     @param leaf is the core representation of a trained leaf.

     @param tIdx is the absolute tree index.

     @param scale estimates a resizing factor.
   */
  void writeNode(const struct TrainChunk* train,
                 unsigned int tIdx,
                 double scale);

  
  /**
   */
  RawVector rawResize(const unsigned char raw[],
                      size_t nodeOff,
                      size_t nodeBytes,
                      double scale);

  /**
     @brief Consumes the BagSample records and writes as raw data.
   */
  void writeBagSample(const struct TrainChunk* train,
                    unsigned int treeOff,
                    double scale);
public:
  vector<size_t> nodeHeight;  // Accumulated per-tree extent of Leaf vector.
  RawVector nodeRaw; // Packed node structures as raw data.

  vector<size_t> bagHeight; // Accumulated per-tree extent of BagSample vector.
  RawVector blRaw; // Packed bag/sample structures as raw data.

  /**
     @brief Constructor.

     @param nTree is the number of trees over which to train.
   */
  LBTrain(unsigned int nTree);

  virtual ~LBTrain() {}
  
  /**
     @brief Static initialization.

     @param thin_ indicates whether certain annotations may be omitted.
   */
  static void init(bool thin_);

  /**
     @brief Resets static initializations.
   */
  static void deInit();


  /**
     @brief High-level entry for writing contents of a tree's leaves.

     @param leaf is the core represenation of the trained leaves.

     @param tIdx is the absolute index of the tree.

     @param scale estimates a resizing factor.
   */
  virtual void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);

  /**
     @brief Packages contents for storage by front end.

     @return named list of summary fields.
   */
  virtual List wrap() = 0;
};


struct LBTrainReg : public LBTrain {
  const NumericVector yTrain; // Training response.

  LBTrainReg(const NumericVector& yTrain_,
             unsigned int nTree);

  ~LBTrainReg() {}

  /**
     @brief Description and parameters as with virutal declaration.
   */
  void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);
  /**
     @brief Description as with virtual declaration.s
   */
  List wrap();
};


/**
   @brief Specialization for categorical leaves, which maintain an
   additional field for weights.
 */
struct LBTrainCtg : public LBTrain {
  NumericVector weight; // Per-category probabilities.
  R_xlen_t weightSize; // Running Size of weight vector.  Not saved.
  const IntegerVector yTrain; // Training response.

  LBTrainCtg(const IntegerVector& yTrain_,
             unsigned int nTree);

  ~LBTrainCtg() {}

  /**
     @brief Description and parameters as with virtual declaration.
   */
  void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);

  /**
     @brief Description as with virtual declaration.
   */
  List wrap();


    /**
   */
  NumericVector numericResize(const double num[],
                          size_t nodeOff,
                          size_t elts,
                          double scale);

private:
  /**
     @brief Writes leaf weights from core representation.

     Not jagged, so tree index parameter unneeded.

     @param leaf is the core representation of a tree's leaves.

     @double scale estimates a resizing factor.
   */
  void writeWeight(const struct TrainChunk* train,
                   double scale);

};

#endif
