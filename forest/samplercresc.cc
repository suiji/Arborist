
#include "samplercresc.h"

SamplerCresc::SamplerCresc(const vector<double>& yNum,
			   unsigned int treeChunk) :
  samplerNux(vector<SamplerNux>(0)),
  yProxy(vector<double>(0)),
  leaf(Leaf::factoryReg(yNum)),
  height(vector<size_t>(treeChunk)) {
}


SamplerCresc::SamplerCresc(const vector<PredictorT>& yCtg,
			   PredictorT nCtg,
			   const vector<double>& yProxy,
			   unsigned int treeChunk) :
  samplerNux(vector<SamplerNux>(0)),
  yProxy(yProxy),
  leaf(Leaf::factoryCtg(yCtg, nCtg)),
  height(vector<size_t>(treeChunk)) {
}


void SamplerCresc::rootSample(const class TrainFrame* frame) {
  sample = leaf->rootSample(frame, yProxy);
}


Sample* SamplerCresc::getSample() const {
  return sample.get();
}


vector<double> SamplerCresc::bagLeaves(const vector<IndexT>& leafMap, unsigned int tIdx) {
  IndexT sIdx = 0;
  for (auto leafIdx : leafMap) {
    samplerNux.emplace_back(sample->getDelRow(sIdx), leafIdx, sample->getSCount(sIdx));
    sIdx++;
  }
  height[tIdx] = samplerNux.size();

  return leaf->scoreTree(sample.get(), leafMap);
}

void SamplerCresc::dumpRaw(unsigned char blRaw[]) const {
  for (size_t i = 0; i < samplerNux.size() * sizeof(SamplerNux); i++) {
    blRaw[i] = ((unsigned char*) &samplerNux[0])[i];
  }
}
