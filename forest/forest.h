// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.h

   @brief Data structures and methods for constructing and walking
       the decision trees.

   @author Mark Seligman
 */

#ifndef FOREST_FOREST_H
#define FOREST_FOREST_H

#include <vector>

#include "decnode.h"
#include "typeparam.h"

/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const DecNode* treeNode;
  const vector<vector<size_t>> leafNode; // Per-tree vector of leaf indices.

  unique_ptr<class BVJaggedV> facSplit; // Consolidation of per-tree values.

  /**
     @brief Collects leaf nodes, by tree.
   */
  vector<vector<size_t>> leafNodes(size_t nNode,
				   unsigned int nTree) const;

  /**
     @brief Enumerates accumulated node heights, by tree.
   */
  vector<size_t> treeHeights() const;


  void dump(vector<vector<PredictorT> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexT> > &lhDelTree) const;

  
 public:

  Forest(size_t forestHeight,
         const DecNode _treeNode[],
         unsigned int facVec[],
         const vector<size_t>& facHeight);

  ~Forest();

  /**
     @brief Getter for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int getNTree() const {
    return leafNode.size();
  }

  /**
     @brief Getter for node records.

     @return pointer to base of node vector.
   */
  inline const DecNode* getNode() const {
    return treeNode;
  }

  
  /**
     @brief Accessor for split encodings.

     @return pointer to base of split-encoding vector.
   */
  inline const BVJaggedV* getFacSplit() const {
    return facSplit.get();
  }
  

  /**
     @brief Derives tree origins from the forest height vector
     and caches.

     @return vector of per-tree node starting offsets.
   */
  vector<size_t> treeOrigins() const;

  
  /**
     @return per-tree vector of leaf scores.
   */
  vector<vector<double>> getScores() const;
  

  /**
     @brief Dumps forest-wide structure fields as per-tree vectors.
     
     Suitable for bridge-level diagnostic methods.

     @param[out] predTree outputs per-tree splitting predictors.

     @param[out] splitTree outputs per-tree splitting criteria.

     @param[out] lhDelTree outputs per-tree lh-delta values.

     @param[out] facSplitTree outputs per-tree factor encodings.
   */
  void dump(vector<vector<PredictorT> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexT> > &lhDelTree,
            vector<vector<PredictorT> > &facSplitTree) const;
};
#endif
