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

#ifndef CART_FOREST_H
#define CART_FOREST_H

#include <vector>

#include "decnode.h"
#include "typeparam.h"

/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const IndexT* nodeHeight;
  const unsigned int nTree;
  const DecNode* treeNode;
  unique_ptr<class BVJagged> facSplit; // Consolidation of per-tree values.

  void dump(vector<vector<PredictorT> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexT> > &lhDelTree) const;

  
 public:

  Forest(const IndexT height_[],
         unsigned int _nTree,
         const DecNode _treeNode[],
         PredictorT _facVec[],
         const IndexT facHeight_[]);

  ~Forest();

  /**
     @brief Getter for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
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
  inline const BVJagged* getFacSplit() const {
    return facSplit.get();
  }
  
  /**
     @brief Determines height of individual tree height.

     @param tIdx is the tree index.

     @return Height of tree.
   */
  inline size_t getNodeHeight(unsigned int tIdx) const {
    return nodeHeight[tIdx];
  }


  /**
     @brief Derives tree origins from the forest height vector
     and caches.

     @return vector of per-tree node starting offsets.
   */
  vector<size_t> cacheOrigin() const;

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
