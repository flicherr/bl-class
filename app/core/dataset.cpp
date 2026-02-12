#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include "dataset.h"

std::vector<Sample> Dataset::load_from_file(const std::string &path)
{
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("Cannot open dataset file");
	}

	std::vector<Sample> data;
	std::string line;

	if (!std::getline(file, line)) {
		throw std::runtime_error("Empty CSV file");
	}

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string token;
		Sample s;

		// x1
		if (!std::getline(ss, token, ',')) continue;
		s.x1 = std::stod(token);

		//x2
		if (!std::getline(ss, token, ',')) continue;
		s.x2 = std::stod(token);

		// y
		if (!std::getline(ss, token, ',')) continue;
		s.y = std::stod(token);

		if (s.y != 1 && s.y != -1) {
			throw std::runtime_error("Label must be +1 or -1");
		}

		data.push_back(s);
	}

	return data;
}

DatasetSplit Dataset::split(
	const std::vector<Sample> &data,
	double train_ratio,
	unsigned seed
) {
	std::vector<Sample> shuffled = data;

	std::mt19937 rng(seed);
	std::shuffle(shuffled.begin(), shuffled.end(), rng);

	size_t train_size = static_cast<size_t>(train_ratio * shuffled.size());

	DatasetSplit split;
	split.train.assign(shuffled.begin(), shuffled.begin() + train_size);
	split.test.assign(shuffled.begin() + train_size, shuffled.end());

	return split;
}
