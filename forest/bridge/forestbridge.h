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
using namespace std;

/**
   @brief Hides class Forest internals from bridge via forward declarations.
 */
struct ForestBridge {

  /**
     @brief Constructor wraps constant raw pointers provided by front end.

     It is the responsibility of the front end and/or its bridge to ensure
     that aliased memory remains live.

     @param height is the accumulated tree height preceding a given tree.

     @param facSplit contains the splitting bits for factors.

     @param facHeight is the accumated factor splitting height.

   */
  ForestBridge(unsigned int nTree,
	       const unsigned char* node,
               unsigned char* facSplit);


  /**
     @brief Training constructor.
   */
  ForestBridge(unsigned int treeChunk);

  
  ~ForestBridge();


  /**
     @brief Passes through to core method.

     @return # bytes subsumed by this chunk of nodes.
   */
  size_t getNodeBytes() const;

  /**
     @brief Returns size of CartNode.
   */
  static size_t nodeSize();

  
  /**
     @brief Returns pointer to core-level Forest.
   */
  class Forest* getForest() const;


  /**
     @brief Getter for tree count;
   */
  unsigned int getNTree() const;


  /**
     @brief Passes through to Forest method.

     @return # bytes in current chunk of factors.
   */
  size_t getFactorBytes() const;
  

  void dumpTreeRaw(unsigned char treeOut[]) const;


  void dumpFactorRaw(unsigned char facOut[]) const;


  /**
     @brief Dumps the forest into per-tree vectors.
   */
  void dump(vector<vector<unsigned int> >& predTree,
            vector<vector<double> >& splitTree,
            vector<vector<unsigned int> >& lhDelTree,
            vector<vector<unsigned int> >& facSplitTree) const;
  
private:

  unique_ptr<class Forest> forest; // Core-level instantiation.
};

#endif
