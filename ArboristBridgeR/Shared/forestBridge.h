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

#include "forest.h"

class ForestBridge : public Forest {
  // References to front end-style vectors, pinned to preserve scope:
  const IntegerVector &feOrigin;
  const RawVector &feFacSplit;
  const IntegerVector &feFacOrig;
  const RawVector &feNode;

  static List Legal(SEXP sForest);
  
 public:
  ForestBridge(const IntegerVector &_feOrigin,
	       const RawVector &_feFacSplit,
	       const IntegerVector &_feFacOrig,
	       const RawVector &_feNode);
  
  /**
     @brief Factory incorporating trained forest cached by front end.

     @param sForest is an R-stye List node containing forest vectors.

     @return bridge specialization of Forest prediction type.
  */
  static unique_ptr<ForestBridge> Unwrap(SEXP sForest);

  static List Wrap(const ForestTrain *forest);
};


class ForestExport {
  const Forest *forest;
  unsigned int nTree;
  vector<vector<unsigned int> > predTree;
  vector<vector<unsigned int> > bumpTree;
  vector<vector<double > > splitTree;
  vector<vector<unsigned int> > facSplitTree;

  void PredExport(const int predMap[]);
  void PredTree(const int predMap[],
	   vector<unsigned int> &pred,
	   const vector<unsigned int> &bump);
  
 public:
  ForestExport(SEXP sForest, IntegerVector &predMap);

  unsigned int NTree() {
    return nTree;
  }
  
  vector<vector<unsigned int> > &PredTree() {
    return predTree;
  }

  vector<vector<unsigned int> > &BumpTree() {
    return bumpTree;
  }

  vector<vector<double> > &SplitTree() {
    return splitTree;
  }

  vector<vector<unsigned int> > &FacSplitTree() {
    return facSplitTree;
  }
};


#endif
