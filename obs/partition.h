// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file partition.h

   @brief Observation frame, partitioned by tree node.

   @author Mark Seligman
 */

#ifndef OBS_PARTITION_H
#define OBS_PARTITION_H


#include "stagedcell.h"
#include "path.h"
#include "typeparam.h"

#include <vector>

#include "obs.h" // Temporary


/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class ObsPart {
  // ObsPart appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.


  // Predictor-based sample orderings, double-buffered by level value.
  //
  const IndexT bagCount;
  const IndexT bufferSize; // <= nRow * nPred.

  Obs* obsCell;

  // 'indexBase' could be boxed with Obs.  While it is used in
  // both replay and restaging, though, it plays no role in splitting.
  // Maintaining a separate vector permits a 16-byte stride to be
  // used for splitting.  More significantly, it reduces memory
  // traffic incurred by transposition on the coprocessor.
  //
  IndexT* indexBase;

 protected:
  //  vector<unsigned int> destRestage;
  //  vector<unsigned int> destSplit; // Coprocessor restaging.
  vector<IndexRange> stageRange; // Index range for staging.
  
  
 public:

  ObsPart(const class PredictorFrame* frame, IndexT bagCount_);

  virtual ~ObsPart();


  /**
     @brief Passes through to bufferOff() using definition coordinate.
   */
  IndexT* getIdxBuffer(const class SplitNux& nux) const;


  Obs* getBuffers(const class SplitNux& nux, IndexT*& sIdx) const;


  Obs* getPredBase(const class SplitNux& nux) const;


  IndexT getSampleIndex(const class SplitNux& cand,
			IndexT obsIdx) const;

  
  IndexT getBagCount() const {
    return bagCount;
  }


  /**
     @brief Sets the staging range for a given predictor.
   */
  void setStageRange(PredictorT predIdx,
		     const IndexRange& safeRange) {
    stageRange[predIdx] = safeRange;
  }
  

  /**
     @brief Returns the staging position for a dense predictor.
   */
  auto getStageOffset(PredictorT predIdx) const {
    return stageRange[predIdx].idxStart;
  }

  // The category could, alternatively, be recorded in an object subclassed
  // under class ObsPart.  This would require that the value be restaged,
  // which happens for all predictors at all splits.  It would also require
  // that distinct ObsPart classes be maintained for SampleReg and
  // SampleCtg.  Recomputing the category value on demand, then, seems an
  // easier way to go.
  //

  /**
     @brief Toggles between positions in workspace double buffer, by level.

     @return workspace starting position for this level.
   */
  IndexT buffOffset(unsigned int bufferBit) const {
    return (bufferBit & 1) == 0 ? 0 : bufferSize;
  }

  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  IndexT bufferOff(PredictorT predIdx, unsigned int bufBit) const {
    return stageRange[predIdx].idxStart + buffOffset(bufBit);
  }


  IndexT bufferOff(const StagedCell* mrra,
			  bool comp = false) const {
    return bufferOff(mrra->getPredIdx(), comp ? mrra->compBuffer() : mrra->bufIdx);
  }

  
  IndexT* idxBuffer(const StagedCell* ancestor) const {
    return indexBase + bufferOff(ancestor);
  }

  
  /**
   */
  Obs* buffers(PredictorT predIdx,
		      unsigned int bufBit,
		      IndexT*& sIdx) const {
    IndexT offset = bufferOff(predIdx, bufBit);
    sIdx = indexBase + offset;
    return obsCell + offset;
  }


  IndexT* indexBuffer(const StagedCell* mrra) const {
    return indexBase + bufferOff(mrra->getPredIdx(), mrra->bufIdx);
  }


  Obs* buffers(const StagedCell* mrra,
	       IndexT*& sIdx) const {
    return buffers(mrra->getPredIdx(), mrra->bufIdx, sIdx);
  }


  const Obs* getSourceBuffer(const StagedCell& mrra) {
    return obsCell + bufferOff(mrra.getPredIdx(), mrra.bufIdx);
  }


  Obs* getPredBase(const StagedCell* mrra) const {
    return obsCell + bufferOff(mrra);
  }
  
  /**
     @brief Returns buffer containing splitting information.
   */
  Obs* Splitbuffer(PredictorT predIdx, unsigned int bufBit) {
    return obsCell + bufferOff(predIdx, bufBit);
  }


  void buffers(const StagedCell& mrra,
		      Obs*& source,
		      IndexT*& sIdxSource,
		      Obs*& targ,
		      IndexT*& sIdxTarg) {
    source = buffers(mrra.getPredIdx(), mrra.bufIdx, sIdxSource);
    targ = buffers(mrra.getPredIdx(), mrra.compBuffer(), sIdxTarg);
  }


  void indexBuffers(const StagedCell* mrra,
                           IndexT*& sIdxSource,
                           IndexT*& sIdxTarg) {
    sIdxSource = indexBase + bufferOff(mrra);
    sIdxTarg = indexBase + bufferOff(mrra, true);
  }


  /**
     @brief Stable partition of observation and index.
   */
  void restageDiscrete(const PathT* prePath,
		       const StagedCell& mrra,
		       vector<IndexT>& obsScatter);


  /**
     @brief As above, but also tracks tied values.
   */
  void restageTied(const PathT* prePath,
		   vector<IndexT>& runCount,
		   const StagedCell& mrra,
		   vector<IndexT>& obsScatter);


  void restageValues(const PathT* prePath,
		     vector<IndexT>& runCount,
		     const StagedCell& mrra,
		     vector<IndexT>& obsScatter,
		     vector<IndexT>& valScatter,
		     const vector<IndexT>& runValue,
		     vector<IndexT>& ranks);
};

#endif
