// Copyright (C)  2012-2019  Mark Seligman
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
   @file exportRf.h

   @brief C++ class definitions for managing class export serializtion.

   @author Mark Seligman

 */


#ifndef RF_EXPORT_RF_H
#define RF_EXPORT_RF_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP Export(SEXP sTrain);

struct ExportRf {

  static List exportLeafReg(const struct LeafExportReg* leaf,
                            unsigned int tIdx);

  static List exportLeafCtg(const struct LeafExportCtg* leaf,
                            unsigned int tIdx);

  static List exportForest(const class ForestExport* forestExport,
                           unsigned int tIdx);

  static IntegerVector exportBag(const struct LeafExport* leaf,
                                 unsigned int tIdx,
                                 unsigned int rowTrain);

  static List exportTreeReg(const List& sTrain,
                            const IntegerVector& predMap);

  static List exportTreeCtg(const class ForestExport* forest,
                            const struct LeafExportCtg* leaf,
                            unsigned int rowTrain);

  static List exportReg(const List& sTrain,
                        const IntegerVector& predMap,
                        const List& predLevel);

  static List exportCtg(const List& sTrain,
                        const IntegerVector& predMap,
                        const List& predLevel);
};

struct LeafExport {
  LeafExport(unsigned int nTree_);

  virtual ~LeafExport() {}

  /**
     @brief Accessor for per-tree sampled row vector.

     @param tIdx is the tree index.

     @return sampled row vector.
   */
  const vector<size_t> &getRowTree(unsigned int tIdx) const {
    return rowTree[tIdx];
  }


  /**
     @brief Accessor for per-tree sample-count vector.

     @param tIdx is the tree index.

     @return sample-count vector.
   */
  const vector<unsigned int> &getSCountTree(unsigned int tIdx) const {
    return sCountTree[tIdx];
  }


  /**
     @brief Accessor for per-tree extent vector.

     @param tIdx is the tree index.

     @return extent vector.
   */
  const vector<unsigned int> &getExtentTree(unsigned int tIdx) const {
    return extentTree[tIdx];
  }

 protected:
  unsigned int nTree;
  vector<vector<size_t> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
};


struct LeafExportReg : public LeafExport {
  /**
    @brief Constructor for export, no prediction.
   */
  LeafExportReg(const List& lLeaf,
                const struct BagBridge* bagBridge);

  ~LeafExportReg() {}

  /**
     @brief Builds bridge object from wrapped front-end data.
   */
  static unique_ptr<LeafExportReg> unwrap(const List &lTrain,
                                          const struct BagBridge* bagBridge);

  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }

 private:
  const NumericVector yTrain;
  vector<vector<double > > scoreTree;
};


struct LeafExportCtg : public LeafExport {
  /**
     @brief Constructor for export; no prediction.
   */
  LeafExportCtg(const List& lLeaf,
                const struct BagBridge* bagBridge);

  ~LeafExportCtg() {}

  static unique_ptr<LeafExportCtg> unwrap(const List &lTrain,
                                          const struct BagBridge* bagBridge);

  
  /**
     @brief Accessor exposes category name strings.

     @return level names vector.
   */
  const CharacterVector &getLevelsTrain() const {
    return levelsTrain;
  }


  /**
     @brief Accessor for per-tree score vector.

     @param tIdx is the tree index.

     @return score vector.
   */
  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }


  /**
     @brief Accessor for per-tree weight vector.

     @param tIdx is the tree index.

     @return weight vector.
   */
  const vector<double> &getWeightTree(unsigned int tIdx) const {
    return weightTree[tIdx];
  }


 private:
  const CharacterVector levelsTrain; // Pinned for summary reuse.
  vector<vector<double > > scoreTree;
  vector<vector<double> > weightTree;
};


#endif
