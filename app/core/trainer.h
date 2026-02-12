#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include "sample.h"
#include "perceptron.h"

class Trainer
{
public:
	void train(
		IPerceptron &model,
		std::vector<Sample> data,
		int max_epochs,
		bool shuffle = true
	);

	double test(
		IPerceptron& model,
		const std::vector<Sample>& data
	);

	std::vector<Weights> history;
};

#endif //TRAINER_H
