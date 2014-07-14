/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "node.h"

Node *Node::node = 0;
int *Node::sample2Node = 0;
int Node::nodeMax = -1;
int Node::probSize = -1;
int Node::totNodes = -1;

Sample *NodeReg::sample = 0;
Node *NodeReg::node = 0;
int *NodeReg::sample2Rank = 0;

NodeCtg *NodeCtg::node = 0;
int NodeCtg::ctgWidth = 0;
int *NodeCtg::yCtg = 0;
SampleCtg *NodeCtg::sampleCtg = 0;
double *NodeCtg::ctgSum = 0;

int NodeCache::minHeight = -1;
int NodeCache::liveCount = -1;
NodeCache *NodeCache::nodeCache = 0;
// Hack:  passes back a request for auxialliary random variate vector of size 'auxRvSize'.
//
void Node::Factory(int _nSamp, int _totNodes, const int _minHeight, int &auxRvSize) {
  totNodes = _totNodes;
  nSamp = _nSamp;
  unsigned twoL = 1; // 2^ #(levels-1)
  
  unsigned uN = nSamp;
  int balancedDepth = 1;
  while (twoL <= uN) { // Next power of two greater than 'nSamp'.
    balancedDepth++;
    twoL <<= 1;
  }

  // There could be as many as (bagCount - 1)/2 levels, in the case of a completely left-
  // or right-leaning tree.
  // Two greater than balanced tree height is empirically well-suited to regression
  // trees.
  // TODO:  Categorical trees may require "unlimited" depth.
  //
  //  if (totNodes == 0)
  //totNodes = balancedDepth + 1; 

  
  // TODO:  Take a harder look.  Reallocation can remedy potential for overflow.
  nodeMax = 1 << (accumExp >= (balancedDepth - 5) ? accumExp : balancedDepth - 5);
  NodeCache::Factory(nodeMax, _minHeight);

  // Initial estimate.  Must be wide enough to be visited by every accumulator/predictor
  // combination at every level, so reallocation check is done at the end of every
  // level.
  //
  probSize = nodeMax * (balancedDepth + 1) * Predictor::NPred();

  PreTree::Factory(nSamp);
  Response::NodeAccFactory(auxRvSize);
  SplitSig::Factory(nodeMax);
}

NodeCache::Factory(int _minHeight) {
  minHeight = _minHeight;
  nodeCache = new NodeCache[nodeMax];
}

// Updates 'nodeMax' and data structures depending upon it.
//
void Node::ReFactory(int _nodeMax, int liveCount) {
  nodeMax = _nodeMax;

  NodeCache::ReFactory(liveCount);

  // Nodes' methods invoke NodeElt methods.
  // These, in turn, realloc all derived arrays, including levelSSFac/Num, as well
  //   as FacRun's data structures.
  //
  node->ReFactory();

  // SplitSig:  levelSSNum/Fac, levelWSFacBits, treeSplitBits:
  SplitSig::ReFactory(nodeMax);
  Train::accumRealloc++; // Tracks reallocations for diagnostics.
}

void NodeCache::ReFactory(int liveCount) {
  NodeCache *temp = new NodeCache[nodeMax];
  for (int i = 0; i < liveCount; i++)
    temp[i] = nodeCache[i];

  delete [] nodeCache;
  nodeCache = temp;
}

//
void NodeReg::Factory() {
  //  cout << Train::nSamp << ", " << maxWidth << ", " << maxReachable << endl;
  // Only need to be as wide as 'bagCount' high watermark over trees.
  //
  sample = new Sample[nSamp];
  sample2Rank = new int[nSamp];

  node = new NodeReg[nodeMax];
  PredNode::FactoryReg(nSamp, nodeMax);
}

// N.B.:  Assumes 'nodeMax' has been reset upstream.
//
void NodeReg::ReFactory() {
  delete [] node;
  node = new NodeReg[nodeMax];
  PredNode::ReFactoryReg(nodeMax);
}

void NodeReg::DeFactory() {
  delete [] sample;
  delete [] sample2Rank;
  delete [] node;
  sample = 0;
  sample2Rank = 0;
  node = 0;

  Node::DeFactory();
  PredNode::DeFactoryReg();
}

void NodeCtg::DeFactory() {
  delete [] node;
  delete [] sampleCtg;
  delete [] NodeCtg::ctgSum;

  Node::DeFactory();
  PredNode::DeFactoryCtg();
  ctgWidth = -1;
  sampleCtg = 0;
}


void Node::DeFactory() {
  node->DeFactory();
  nodeMax = -1;
  probSize = -1;
  totNodes = -1;

  NodeCache::DeFactory();
  PreTree::DeFactory();
}

void NodeCache::DeFactory() {
  delete [] nodeCache;
  nodeCache = 0;
}

void NodeCtg::Factory(int _yCtg[], int _ctgWidth, int &auxSize) {
  sampleCtg = new SampleCtg[nSamp];
  yCtg = _yCtg;
  ctgWidth =_ctgWidth;
  node = new NodeCtg[nodeMax];
  NodeCtg::ctgSum = new double[nodeMax * ctgWidth];
  auxSize = PredNode::FactoryCtg(nSamp, nodeMax, _ctgWidth);
}

// 'auxSize' unchanged.
//
void NodeCtg::ReFactory() {
  delete [] node;
  delete [] ctgSum;
  node = new NodeCtg[nodeMax];
  ctgSum = new double[nodeMax * ctgWidth];
  PredNode::ReFactoryCtg(nodeMax);
}

void NodeReg::SetPrebias(const int liveIdx) {
  preBias = (sum * sum) / sCount;
}

  /* ASSERTIONS:
  if (abs(sum - sCount) > 0.1)
    cout << "Jitter mismatch" << endl;
  double sumByCtg = 0.0;
  for (int i = 0; i < LevelCtg::ctgWidth; i++) {
    sumByCtg += sumBase[i];
  }
  if (abs(sumByCtg - sum) > 0.1)
    cout << sumByCtg - sum << endl;
  */
void NodeCtg::SetPrebias(const int liveIdx) {
  sumSquares = 0.0;
  double *sumBase = ctgSum + liveIdx * ctgWidth;
  for (int i = 0; i < ctgWidth; i++)
    sumSquares += sumBase[i] * sumBase[i];

  // 'sum' is zero iff all categories are empty, so will never make it into the
  // denominator.
  preBias = sumSquares / sum;
}

// Caches all node information from the current level into NodeCache workspace.
// This circumvents crosstalk as the next level's nodes are populated.
//
void NodeCache::CacheNodes(int _liveCount) {
  liveCount = _liveCount;
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    int _idxCount, _sCount;
    double _sum, _preBias;
    SplitNode _par;
    bool _isLH;
    node->CacheFields(liveIdx, _idxCount, _sCount, _sum, _preBias, _par, _isLH);
    nodeCache[liveIdx]->SetNode(_idxCount, _sCount, _sum, _preBias, _par, _isLH);
  }
}


void NodeCache::SetNode(int _lhStart, int _idxCount, int _sCount, double _sum, double _preBias, SplitNode *_par, bool _isLH) {
  lhStart = _lhStart;
  idxCount = _idxCount;
  sCount = _sCount;
  sum = _sum;
  preBias = _preBias;
  par = _par;
  isLH = _isLH;
}

int NodeCache::LHRH1(int liveCount, int level) {
  int countNext = 0;
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++)
    nodeCache[liveIdx].LHRH1(liveIdx, level, countNext);

  return countNext;
}

void NodeCache::LHRH2(int liveCount) {
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++)
    node[liveIdx].LHRH2();
}

// Computes split information and next-level sample2Accum[] values directly
// from "elected" PredOrd[].  Replaces SetLHS/LHRun() and SampleReset().
//
    // ASSERTION:
    //    if (predIdx > Predictor::nPred)
    //cout << "Bad predictor index" << endl;
// With chosen treeOrd[] in hand, all boundaries for next level's descendants
// can be computed in Replay().
//
// Sets up live/leaf nodes at next level.
// Node does split, so LHS/RHS checked for size constraints.  There must be at least
// 'minHeight' samples subsumed by the either node, as well as two or more nodes.
//
// N.B.:  These nodes cannot exist as accumulators until LHRH2().  Dispatching as
// leaves is safe, provided that the correct 'idxCount', 'sum' and 'sCount' values are known.
//
void NodeCache::LHRH1(int liveIdx, int level, int &countNext) {
  int predIdx = SplitSig::ArgMaxGini(liveCount, liveIdx, preBias, par == 0 ? 0.0 : par->Gini, lhIdxCount, lhSCount);
  if (predIdx < 0) {
    PreTree::AddLeaf(this);
    // Any SampleOrd[], such as that for predIdx == 0, presents a valid list of sample indices.
    (void) Node::SampleReplay(0, level, lhStart, idxCount, leafOrPred);
  }
  else {
    leafOrPred = predIdx;
    preTree = PreTree::AddSplit(predIdx, preBias, liveIdx, par, isLH);

    if (TerminalLeft(lhSCount, lhIdxCount))
      lhId = PreTree::AddLeaf(preTree, true);
    else
      lhId = countNext++;

    if (TerminalRight(lhSCount, lhIdxCount))
      rhId = PreTree::AddLeaf(preTree, false);
    else // RHS perists
      rhId = countNext++;

    lhSum = SplitSig::LHRH(predIdx, liveIdx, level, lhId, rhId);
  }
}

// Two-sided SampleReplay(), called for numeric SplitSigs, for which only the left-hand
// index, 'lhEdge', is known.
//
double NodeCache::SampleReplayLHRH(int liveIdx, int predIdx, int level, int lhCount) {
  NodeCache *stt = &nodeCache[liveIdx];
  int start = stt->lhStart;
  double lhSum = Node::SampleReplay(predIdx, level, start, lhCount, stt->lhId);
  (void) Node::SampleReplay(predIdx, level, start + lhCount, stt->idxCount - lhCount, stt->rhId);

  return lhSum;
}

// Single-sided SampleReplay().
//
double Node::SampleReplay(int predIdx, int level, int start, int count, int id) {
  return node->SampleReplay(predIdx, level, start, count, id);
}

// Finishes leaves and initializes accumulators for next level.  Accumulator initialization
// requires LHS sum values, so cannot occur until after sample vector is walked for
// "winning" PredOrd.
//
void NodeCache::LHRH2() {
  if (leafOrPred < 0)
    return;

  if (lhId >= 0)
    node->EarlyFields(lhId, lhIdxCount, lhSCount, true, lhSum, preTree);
  else
    PreTree::LeafComplete(lhId, lhIdxCount, lhSum, lhSCount);

  int rhIdxCount = idxCount - lhIdxCount;
  double rhSum = sum - lhSum;
  int rhSCount = sCount - lhSCount;
  if (rhId >= 0)
    node->EarlyFields(rhId, rhIdxCount, rhSCount, false, rhSum, preTree);
  else
    PreTree::LeafComplete(rhId, rhIdxCount, rhSum, rhSCount);
}

// Static entry point for this level's restaging.
// 
void NodeCache::Restage(int liveCount, int predIdx, int level) {
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    nodeCache[liveIdx].Restage(predIdx, level);
}

//  Maps into next level's offsets from this level's accumulator indices.  Requires
//  the 'lhStart' field to be set for every accumulator in the next level.
//
void NodeCache::Restage(int predIdx, int level) {
  if (leafOrPred >= 0 && (rhId >= 0 || lhId >= 0))
    node->Restage(predIdx, level, lhStart, idxCount, lhId, rhId);
}


// Conveys splits to next level.
//
// Sets sample2Node[]:
//
//   Accumulators already terminal/negative:  unchanged.
//
//   Accumulators not splitable in this level become terminal:  -(accum+1+twoL-1).
//
//   Accumulators splitable at next level:  LHS unchanged, RHS to 'off' + 'rhOff'.
//
//   RHS/LHS of split accumulators too small to persist become terminal:  -(accum+1+twoL-1).
//
int Node::NextLevel(int liveCount, int level) {
  // TODO:  MUST guarantee that no zero-length "splits" have been introduced.  Not only are
  // these nonsensical, but they are also dangerous, as they violate various assumptions about
  // the integrity of the intermediate respresentation.
  //
  int countNext = NodeCache::LHRH(liveCount, level);

  node->Reset(countNext, sample2Node);
  NextLevel(countNext);

  return countNext;
}

int NodeCache::LHRH(int liveCount, int level) {
  CacheNodes(liveCount);

  // Checks every live accumulator.  If not split, records on leafSet[].  If split, checks
  // LHS and RHS against size constraints.  Child nodes not meeting size constraints are
  // recorded on leafSet[]. Those meeting the constraints are enumerated by 'countNext' as
  // live in the next level.
  //
  int countNext = LHRH1(liveCount, level);

  // Beyond this point, new accumulators must be allocated for next level.  Before
  // this point, various accumulator-derived data structures are still live.
  // Hence this is likely the most practical place to peform reallocation.
  //
  if (countNext > nodeMax)
    Node::ReFactory(nodeMax << 1, liveCount);
  //  cout << countNext << " / " << nodeMax << endl;

  LHRH2(liveCount);

  return countNext;
}

// Rewrites the node set for the next level.
// Accumulates and sets starting offsets, as well as "late" fields.
//
void Node::NextLevel(int countNext) {
  int idx = 0;
  for (int liveIdx = 0; liveIdx < countNext; liveIdx++)
    idx += node->LateFields(liveIdx, idx);
}


void NodeCtg:Reset(int countNext, const int sample2Node[]) {
  // Initializes 'ctgSum' values for accumulators making it to the next level.
  // Since 'countNext' may exceed the accumulator count, this should not take
  // place until any necessary accumulator reallocation has been performed.
  //
  for (int i = 0; i < countNext * ctgWidth; i++)
    ctgSum[i] = 0.0;

  // ctgSum[] must be reset before resetting the 'preBias'.
  //
  for (int i = 0; i < bagCount; i++) {
    int idx = sample2Node[i];
    if (idx >= 0) {
      int ctg = sampleCtg[i].ctg;
      ctgSum[ctgWidth * idx + ctg] += sampleCtg[i].val;
    }
  }
  // Number of category entries should equal 'idxCount' for accumulator.
  // ASSERTION:  TODO
  //
  // Fields 'preBias' and 'lhStart' are set in the Reset pass.
}

// Monolith entry point for per-level splitting.
//
// Returns count of levels.
//
int Node::Levels(int bagCount, const double auxRv[]) {
  node->TreeInit(bagCount, auxRv);

  double *treePredProb = Util::Sample(probSize);
  int nPred = Predictor::NPred();
  int ruCt = 0;
  int liveCount = 1;// Single root node at level zero.
  int level;

  for (level = 0; liveCount > 0 && (totLevels == 0 ||  level < totLevels); level++) {
    Train::LevelReset(liveCount);
    PredNode::Level(liveCount, level);
    liveCount = NextLevel(liveCount, level);
  }
  NodeCache::LevelToLeaf(liveCount, bagCount, sample2Node);

  return level + 1;
}

// Prepares for decision-tree scoring:
//   Flushes remaining live nodes to the leaf list for scoring.
//
void NodeCache::LevelToLeaf(int liveCount, int bagCount, int sample2Node[]) {
  CacheNodes(liveCount);
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++)
    PreTree::AddLeaf(&nodeCache[liveIdx]);

  // Reconciles remaining live sample2Node[] values with leaf indices.
  // Clients include categorical response and quantile regression.
  //
  for (int i = 0; i < bagCount; i++) {
    int stIdx = sample2Node[i];
    if (stIdx >= 0)
      stIdx = nodeCache[stIdx].leafOrPred;
    sample2Node[i] = -(1 + stIdx); // Positive offsets into 'leafSet'.
  }
}

// 
// 
void Node::TreeInit(int bagCount) {
  sample2Node = new int[bagCount];
  for (int i = 0; i < bagCount; i++) {
    sample2Node[i] = 0; // Unique root nodes zero.
  }

  PredNode::TreeInit(bagCount);
  PreTree::TreeInit(bagCount);
  SplitSigFac::TreeInit();
}

//
void NodeReg::TreeInit(int bagCount, double *auxRv) {
  Node::TreeInit(bagCount);

  // TODO:  May be unecessary.
  for (int i = bagCount; i < nSamp; i++) {
    sample[i].val = 0.0;
  }

  double sum = 0.0;
  for (int i = 0; i < bagCount; i++)
    sum += sample[i].val;
  node->EarlyFields(0, bagCount, nSamp, false, sum, 0);
  node->LateFields(0, 0);
}

// 'ctgSum' is allocated per-session, so must be reinitialized on tree entry.
//
void NodeCtg::TreeInit(int bagCount, double *auxRv) {
  Node::TreeInit(bagCount);

  // TODO:  May be unecessary.
  for (int i = bagCount; i < nSamp; i++) {
    sampleCtg[i].val = 0.0;
  }

  // Sets 'ctgSum' for all indices pertaining to accumulator zero.
  for (int i = 0; i < ctgWidth; i++)
    ctgSum[i] = 0.0;
  double sum = 0.0;
  for (int i = 0; i < bagCount; i++) {
    int ctg = sampleCtg[i].ctg;
    double sampleSum = sampleCtg[i].val;
    ctgSum[ctg] += sampleSum;
    sum += sampleSum;
  }

  node->EarlyFields(0, bagCount, nSamp, false, sum, 0);
  node->LateFields(0, 0);

  PredCtgFac::TreeInit(auxRv);
}

void Node::ClearTree() {
  delete [] sample2Node;
  sample2Node = 0;
}

void NodeCtg::ClearTree() {
  Node::ClearTree();
  PredCtgFac::ClearTree();
}


// N.B.:  'leafId', below, are negative-encoded offsets into the leaf set.

// Completes the field initialization for leaves from splitting parents, which is delayed
// until the LHS/RHS node elements have been determined.
//
void Node::LeafComplete(Leaf *leaf, int idxCount, double sum, int sCount) {
  leaf->extent = idxCount;
}

// 'score' must be assigned in order to complete a regression leaf.
//
void NodeReg::LeafComplete(Leaf *leaf, int idxCount, double sum, int sCount) {
  Node::LeafComplete(leaf, idxCount, sum, sCount);
  leaf->score = sum / sCount;
}

