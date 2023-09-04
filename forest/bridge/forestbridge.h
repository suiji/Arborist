// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestbridge.h

   @brief Front-end wrappers for core Forest objects.

   @author Mark Seligman
 */

#ifndef FOREST_FORESTBRIDGE_H
#define FOREST_FORESTBRIDGE_H


#include <vector>
#include <memory>
#include <complex>

using namespace std;

/**
   @brief Hides class Forest internals from bridge via forward declarations.
 */
struct ForestBridge {
  
  /**
     @brief R-specific constructor.  Doubles cache large offset values.

     It is the responsibility of the front end and/or its bridge to ensure
     that aliased memory either remains live or is copied.

     @param nTree is the number of tres.
     
     @param treeNode caches the nodes as packed-integer / double pairs.

     @param nPred is the number of training predictors.

     @param scores cache the score at each node, regardless whether terminal.

     @param facExtent the per-tree count of factor-valued splits.

     @param facSplit contains the splitting bits for factors.
   */
  ForestBridge(unsigned int nTree,
	       const double nodeExtent[],
	       const complex<double> treeNode[],
	       const double scores[],
	       const double facExtent[],
               const unsigned char facSplit[],
	       const unsigned char facObserved[],
	       const tuple<double, double, string>& scoreDesc);

  
  /**
     @brief Training constructor.
   */
  //  ForestBridge(unsigned int treeChunk);

  
  ForestBridge(ForestBridge&&);

  
  ~ForestBridge();


  /**
     @brief Initializes Forest statics.
   */
  static void init(unsigned int nPred);


  /**
     @brief Resets Forest statics.
   */
  static void deInit();
  
  
  /**
     @brief Returns pointer to core-level Forest.
   */
  class Forest* getForest() const;


  /**
     @brief Getter for tree count;
   */
  unsigned int getNTree() const;


  const vector<size_t>& getFacExtents() const;


  /**
     @brief Dumps the forest into per-tree vectors.
   */
  void dump(vector<vector<unsigned int> >& predTree,
            vector<vector<double> >& splitTree,
            vector<vector<size_t> >& lhDelTree,
            vector<vector<unsigned char> >& facSplitTree,
	    vector<vector<double>>& scoreTree) const;
  
private:

  unique_ptr<class Forest> forest; ///< Core-level instantiation.
};

#endif
