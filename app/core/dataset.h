#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include "sample.h"

struct DatasetSplit
{
	std::vector<Sample> train;
	std::vector<Sample> test;
};

class Dataset
{
public:
	static std::vector<Sample> load_from_file(const std::string &path);
	static DatasetSplit split(
		const std::vector<Sample> &data,
		double train_ratio = 0.66,
		unsigned seed = 42
	);
};

#endif //DATASET_H
