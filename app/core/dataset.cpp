#include <fstream>
#include <sstream>
#include <stdexcept>
#include "dataset.h"

template <>
std::vector<ClSample> dataset::load_from_file(const std::string &path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open dataset file");
    }

    std::vector<ClSample> data;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file");
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ClSample s;

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

template <>
std::vector<RegSample> dataset::load_from_file(const std::string &path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open dataset file");
    }

    std::vector<RegSample> data;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file");
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        RegSample s{};

        if (!std::getline(ss, token, ',')) continue;
        s.x = std::stod(token);

        if (!std::getline(ss, token, ',')) continue;
        s.y = std::stod(token);

        data.push_back(s);
    }

    return data;
}