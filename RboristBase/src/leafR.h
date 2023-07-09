// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file leafR.h

   @brief C++ interface to R entry for sampled leaves.

   @author Mark Seligman
 */

#ifndef LEAF_R_H
#define LEAF_R_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief Summary of leaf samples.
 */
struct LeafR {
  static const string strExtent;
  static const string strIndex;

  NumericVector extent; ///< Leaf extents.
  NumericVector index; ///< Sample indices.
  size_t extentTop; ///< top of leaf extent buffer.
  size_t indexTop;  ///< " " sample index buffer.


  LeafR();

  /**
     @brief Bundles trained leaf into format suitable for R.

     Wrap functions are called from TrainR::summary, following which 'this' is
     deleted.  There is therefore no need to initialize the extent and index
     state.
     
   */
  List wrap();

  
  /**
     @brief Consumes a block of samples following training.

     @param scale is a fudge-factor for resizing.
   */
  void bridgeConsume(const struct LeafBridge& sb,
		     double scale);


  static LeafBridge unwrap(const List& lLeaf,
			   const struct SamplerBridge& samplerBridge);
};


struct LeafExpand {
  LeafExpand(const List& lSampler);

  virtual ~LeafExpand() {}

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


struct LeafExpandReg : public LeafExpand {
  /**
    @brief Constructor for export, no prediction.
   */
  LeafExpandReg(const List& lSampler);


  ~LeafExpandReg();


  /**
     @brief Builds bridge object from wrapped front-end data.
   */
  static LeafExpandReg unwrap(const List &lTrain);

};


struct LeafExpandCtg : public LeafExpand {
  /**
     @brief Constructor for export; no prediction.
   */
  LeafExpandCtg(const List& lSampler);

  
  ~LeafExpandCtg();

  
  static LeafExpandCtg unwrap(const List &lTrain);

  
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
