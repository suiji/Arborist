// Copyright (C)  2012-2025  Mark Seligman
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
   @file forestRf.H

   @brief Bridge access to core Forest type for Random Forest algorithm.

   @author Mark Seligman

 */

#ifndef FOREST_R_H
#define FOREST_R_H


#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
#include <vector>
using namespace std;

struct SamplerBridge;

/**
   @brief Front-end access to ForestBridge.
 */
struct ForestR {

  /**
     @brief Looks up and verifies forest member.

     @return forest member.
   */
  static List checkForest(const List& lTrain);

  
  /**
     @brief Dumping unwrapper.

     @param sTrain is an R-stye List node containing forest vectors.

     @param categorical indicates classification; legacy support only.
     
     @return bridge specialization of Forest prediction type.
  */
  static struct ForestBridge unwrap(const List& sTrain,
				    bool categorical = false);

  
  /**
     @brief Prediction unwrapper.
   */
  static struct ForestBridge unwrap(const List& sTrain,
				    const SamplerBridge& samplerBridge);


  /**
     @brief Unwraps the score descriptor as a tuple.

     @param categorical is true iff classification:  legacy support.
   */
  static tuple<double, double, string> unwrapScoreDesc(const List& lTrain,
						       bool categorical);
};


/**
   @brief As above, but with additional members to facilitate dumping on
   a per-tree basis.
 */
class ForestExpand {
  vector<vector<unsigned int> > predTree;
  vector<vector<size_t> > bumpTree;
  vector<vector<int>> senseTree;
  vector<vector<double > > splitTree;
  vector<vector<unsigned char> > facSplitTree;
  vector<vector<double>> scoreTree; ///< All nodes have scores.

  void predExport(const int predMap[]);
  void treeExport(const int predMap[],
                  vector<unsigned int>& pred,
                  const vector<size_t>& bump);

 public:
  ForestExpand(const List& forestList,
               const IntegerVector& predMap);

  static ForestExpand unwrap(const List &lTrain,
			     const IntegerVector &predMap);


  /**
     @brief Exportation methods for unpacking per-tree node contents
     as separate vectors.

     @param tIdx is the tree index.

     @return vector of unpacked values.
   */
  const vector<unsigned int>& getPredTree(unsigned int tIdx) const {
    return predTree[tIdx];
  }

  const vector<size_t>& getBumpTree(unsigned int tIdx) const {
    return bumpTree[tIdx];
  }

  const vector<double>& getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }
  

  const vector<double> &getSplitTree(unsigned int tIdx) const {
    return splitTree[tIdx];
  }

  const vector<unsigned char>& getFacSplitTree(unsigned int tIdx) const {
    return facSplitTree[tIdx];
  }


  const vector<vector<int>>& getSenseTree() const {
    return senseTree;
  }


  static List expand(const List& sTrain,
		     const IntegerVector& predMap);


  static List expandTree(const class ForestExpand& forestExpand,
                           unsigned int tIdx);
};


/**
   @brief Accumulates R-style representation of crescent forest during
   training.
 */
struct FBTrain {
  static const string strNTree;
  static const string strNode;
  static const string strExtent;
  static const string strTreeNode;
  static const string strScores;
  static const string strFactor;
  static const string strFacSplit;
  static const string strObserved;
  static const string strScoreDesc;
  static const string strNu;
  static const string strBaseScore;
  static const string strForestScorer;

  const unsigned int nTree; ///< Total # trees under training.

  // Decision node related:
  NumericVector nodeExtent; ///< # nodes in respective tree.
  size_t nodeTop; ///< Next available index in node/score buffers.
  ComplexVector cNode; ///< Nodes encoded as complex pairs.
  NumericVector scores; ///< Same indices as nodeRaw.

  // Factor related:
  NumericVector facExtent; // # factor entries in respective tree.
  size_t facTop; // Next available index in factor buffer.
  RawVector facRaw; // Bit-vector representation of factor splits.
  RawVector facObserved; // " " observed levels.

  // Scoring descriptor:
  double nu; ///< Learning rate.
  double baseScore; ///< Score of sampled root.
  string forestScorer; ///< How to score the forest.

  FBTrain(unsigned int nTree);
  

  /**
     @brief Decorates trained forest for storage by front end.
   */
  List wrap();


  /**
     @brief Copies core representation of forest components.

     @param grove caches a crescent forest chunk.

     @param treeOff is the beginning tree index of the grove.

     @param fraction is a scaling factor used to estimate buffer size.
   */
  void groveConsume(const struct GroveBridge* grove,
		    unsigned int treeOff,
		    double fraction);


  void scoreDescConsume(const struct TrainBridge& trainBridge);
  

private:

  List wrapFactor();

  List wrapNode();
  /**
     @brief Copies core representation of a chunk of trained tree nodes.

     @param bridge caches a crescent forest chunk.

     @param treeOff is the beginning tree index of the trained chunk.

     @param fraction is a scaling factor used to estimate buffer size.
   */
  void nodeConsume(const struct GroveBridge* grove,
		   unsigned int treeOff,
		   double fraction);


  /**
     @brief As above, but collects factor-splitting parameters.
   */
  void factorConsume(const struct GroveBridge* bridge,
		     unsigned int treeOff,
		     double fraction);


  /**
     @brief Summarizes requirements of the training algorithm.
   */
  List summarizeScoreDesc();
};

#endif
