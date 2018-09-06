// Copyright (C)  2012-2018  Mark Seligman
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
   @file forestBridge.h

   @brief Bridge specializaton of Core Forest type.

   @author Mark Seligman

 */


#ifndef ARBORIST_FOREST_BRIDGE_H
#define ARBORIST_FOREST_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
#include <vector>
using namespace std;

/**
   @brief Instantiates trained forest for prediction.
 */
class ForestBridge {
  // References to front end-style vectors:  can be pinned to preserve scope:
  const IntegerVector &feOrigin;
  const RawVector &feFacSplit;
  const IntegerVector &feFacOrig;
  const RawVector &feNode;

protected:
  static SEXP Legal(const List &lForest);
  unique_ptr<class Forest> forest;
  
 public:
  ForestBridge(const IntegerVector &_feOrigin,
               const RawVector &_feFacSplit,
               const IntegerVector &_feFacOrig,
               const RawVector &_feNode);


  const class Forest *getForest() const {
    return forest.get();
  }


  const unsigned int getNTree() const {
    return feOrigin.length();
  }


  /**
     @brief Factory incorporating trained forest cached by front end.

     @param sTrain is an R-stye List node containing forest vectors.

     @return bridge specialization of Forest prediction type.
  */
  static unique_ptr<ForestBridge> unwrap(const List &sTrain);
};


/**
   @brief As above, but with additional members to facilitate dumping on
   a per-tree basis.
 */
class ForestExport final : public ForestBridge {
  vector<vector<unsigned int> > predTree;
  vector<vector<unsigned int> > bumpTree;
  vector<vector<double > > splitTree;
  vector<vector<unsigned int> > facSplitTree;

  void PredExport(const int predMap[]);
  void PredTree(const int predMap[],
           vector<unsigned int> &pred,
           const vector<unsigned int> &bump);

 public:
  ForestExport(List &forestList,
               IntegerVector &predMap);

  static unique_ptr<ForestExport> unwrap(const List &lTrain,
                                         IntegerVector &predMap);

  /**
     @brief Exportation methods for unpacking per-tree node contents
     as separate vectors.

     @param tIdx is the tree index.

     @return vector of unpacked values.
   */
  const vector<unsigned int> &getPredTree(unsigned int tIdx) const {
    return predTree[tIdx];
  }

  const vector<unsigned int> &getBumpTree(unsigned int tIdx) const {
    return bumpTree[tIdx];
  }

  const vector<double> &getSplitTree(unsigned int tIdx) const {
    return splitTree[tIdx];
  }

  const vector<unsigned int> &getFacSplitTree(unsigned int tIdx) const {
    return facSplitTree[tIdx];
  }
};


/**
   @brief Accumulates R-style representation of crescent forest during
   training.
 */
struct FBTrain {
  RawVector nodeRaw; // Packed representation of decision tree.
  RawVector facRaw; // Bit-vector representation of factor splits.
  R_xlen_t nodeOff;
  R_xlen_t facOff;

  IntegerVector origin;
  IntegerVector facOrigin;

  FBTrain(unsigned int nTree);

  void consume(const class ForestTrain* forest,
               unsigned int treeOff,
               double fraction);

  List wrap();
};

#endif
