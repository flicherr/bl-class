add_library(core
        dataset.cpp
        neurons/regression_neuron.cpp
        neurons/classifier_neuron_by_threshold.cpp
        neurons/classifier_neuron_by_bias.cpp
        trainers/regression_trainer.cpp
        trainers/classifier_trainer.cpp
)
target_include_directories(core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/neurons
        ${CMAKE_CURRENT_SOURCE_DIR}/trainers
)
target_link_libraries(core PUBLIC)