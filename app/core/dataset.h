#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <random>
#include <algorithm>
#include "samples.h"

namespace dataset
{
template<typename TSample>
std::vector<TSample> load_from_file(const std::string &path);

template<> std::vector<ClSample>  load_from_file<ClSample> (const std::string&);
template<> std::vector<RegSample> load_from_file<RegSample>(const std::string&);

template<typename TSample>
struct DatasetSplit
{
	std::vector<TSample> train;
	std::vector<TSample> test;
};

template<typename TSample>
static DatasetSplit<TSample> split(
	const std::vector<TSample> &data,
	double train_ratio = 0.66,
	unsigned seed = 42)
{
	std::vector<TSample> shuffled = data;

	std::mt19937 rng(seed);
	std::shuffle(shuffled.begin(), shuffled.end(), rng);

	size_t train_size = static_cast<size_t>(train_ratio * shuffled.size());

	DatasetSplit<TSample> split;
	split.train.assign(shuffled.begin(), shuffled.begin() + train_size);
	split.test.assign(shuffled.begin() + train_size, shuffled.end());

	return split;
}
}

#endif //DATASET_H
