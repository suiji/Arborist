// Copyright (C)  2012-2022  Mark Seligman
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
   @file exportR.h

   @brief Expands trained forest into a collection of vectors.

   @author Mark Seligman
 */


#ifndef RF_EXPORT_R_H
#define RF_EXPORT_R_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP expandRf(SEXP sTrain);

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
                        const List& predLevel,
			const List& predFactor);

  static List exportCtg(const List& sTrain,
                        const IntegerVector& predMap,
                        const List& predLevel);
};

struct LeafExport {
  LeafExport(const List& lSampler);

  virtual ~LeafExport() {}

  //  unique_ptr<class LeafBridge> unwrap(const class SamplerBridge* samplerBridge,
  //				      const List& lLeaf);

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

  
  /**
     @brief Accessor for per-tree score vector.
   */
  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }

 protected:
  unsigned int nTree;
  vector<vector<size_t> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
  vector<vector<double > > scoreTree;
};


struct LeafExportReg : public LeafExport {
  /**
    @brief Constructor for export, no prediction.
   */
  LeafExportReg(const List& lTrain,
		const List& lSampler);


  ~LeafExportReg() {}

  /**
     @brief Builds bridge object from wrapped front-end data.
   */
  static unique_ptr<LeafExportReg> unwrap(const List &lTrain);

};


struct LeafExportCtg : public LeafExport {
  /**
     @brief Constructor for export; no prediction.
   */
  LeafExportCtg(const List& lTrain,
		const List& lSampler);


  ~LeafExportCtg() {}

  static unique_ptr<LeafExportCtg> unwrap(const List &lTrain);

  
  /**
     @brief Accessor exposes category name strings.

     @return level names vector.
   */
  const CharacterVector &getLevelsTrain() const {
    return levelsTrain;
  }

  
 private:
  const CharacterVector levelsTrain; // Pinned for summary reuse.
};


#endif
