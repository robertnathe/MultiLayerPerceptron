#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

class NeuralNetwork {
private:
    size_t inputSize, hiddenSize, outputSize;
    std::vector<double> W1, W2, W3, W4, W5; // Weights for 4 hidden layers
    double learningRate;

    void apply_softmax(std::vector<double>& vec) const {
        double max = *std::max_element(vec.begin(), vec.end());
        double sum = 0.0;
        for (auto& x : vec) {
            x = std::exp(x - max);
            sum += x;
        }
        for (auto& x : vec) x /= sum;
    }

public:
    NeuralNetwork(size_t inSize, size_t hidSize, size_t outSize, double lr)
        : inputSize(inSize), hiddenSize(hidSize), outputSize(outSize), learningRate(lr) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);

        // Initialize weights with proper dimensions for 4 hidden layers
        W1.resize(inputSize * hiddenSize); // input -> hidden1
        W2.resize(hiddenSize * hiddenSize); // hidden1 -> hidden2
        W3.resize(hiddenSize * hiddenSize); // hidden2 -> hidden3
        W4.resize(hiddenSize * hiddenSize); // hidden3 -> hidden4
        W5.resize(hiddenSize * outputSize); // hidden4 -> output

        for (auto& w : W1) w = dist(gen);
        for (auto& w : W2) w = dist(gen);
        for (auto& w : W3) w = dist(gen);
        for (auto& w : W4) w = dist(gen);
        for (auto& w : W5) w = dist(gen);
    }

    // Forward pass with four hidden layers
    std::vector<double> forward(const std::vector<double>& input,
                                             std::vector<double>& hidden1Out,
                                             std::vector<double>& hidden2Out,
                                             std::vector<double>& hidden3Out,
                                             std::vector<double>& hidden4Out) const {
        // Hidden layer 1
        std::vector<double> h1(hiddenSize);
        for (size_t i = 0; i < hiddenSize; ++i) {
            h1[i] = std::inner_product(input.begin(), input.end(),
                                            W1.begin() + i * inputSize, 0.0);
        }
        apply_softmax(h1);
        hidden1Out = h1;

        // Hidden layer 2
        std::vector<double> h2(hiddenSize);
        for (size_t i = 0; i < hiddenSize; ++i) {
            h2[i] = std::inner_product(h1.begin(), h1.end(),
                                            W2.begin() + i * hiddenSize, 0.0);
        }
        apply_softmax(h2);
        hidden2Out = h2;

        // Hidden layer 3
        std::vector<double> h3(hiddenSize);
        for (size_t i = 0; i < hiddenSize; ++i) {
            h3[i] = std::inner_product(h2.begin(), h2.end(),
                                            W3.begin() + i * hiddenSize, 0.0);
        }
        apply_softmax(h3);
        hidden3Out = h3;

        // Hidden layer 4
        std::vector<double> h4(hiddenSize);
        for (size_t i = 0; i < hiddenSize; ++i) {
            h4[i] = std::inner_product(h3.begin(), h3.end(),
                                            W4.begin() + i * hiddenSize, 0.0);
        }
        apply_softmax(h4);
        hidden4Out = h4;

        // Output layer
        std::vector<double> output(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            output[i] = std::inner_product(h4.begin(), h4.end(),
                                             W5.begin() + i * hiddenSize, 0.0);
        }
        apply_softmax(output);

        return output;
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets, size_t epochs) {
        std::vector<double> h1Out(hiddenSize), h2Out(hiddenSize), h3Out(hiddenSize), h4Out(hiddenSize);
        std::vector<double> deltaOut(outputSize), deltaH4(hiddenSize), deltaH3(hiddenSize), deltaH2(hiddenSize), deltaH1(hiddenSize);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double totalError = 0.0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& input = inputs[i];
                int target = targets[i];

                // Forward pass
                auto outputs = forward(input, h1Out, h2Out, h3Out, h4Out);

                // Calculate error
                std::vector<double> targetVec(outputSize, 0.0);
                targetVec[target] = 1.0;
                for (size_t j = 0; j < outputSize; ++j)
                    totalError += 0.5 * pow(targetVec[j] - outputs[j], 2);

                // Backpropagation
                // Output layer delta
                for (size_t j = 0; j < outputSize; ++j)
                    deltaOut[j] = (outputs[j] - targetVec[j]) * outputs[j] * (1.0 - outputs[j]);

                // Hidden layer 4 delta
                for (size_t k = 0; k < hiddenSize; ++k) {
                    double sum = 0.0;
                    for (size_t j = 0; j < outputSize; ++j)
                        sum += W5[k + j * hiddenSize] * deltaOut[j];
                    deltaH4[k] = sum * h4Out[k] * (1.0 - h4Out[k]);
                }

                // Hidden layer 3 delta
                for (size_t k = 0; k < hiddenSize; ++k) {
                    double sum = 0.0;
                    for (size_t j = 0; j < hiddenSize; ++j)
                        sum += W4[k + j * hiddenSize] * deltaH4[j];
                    deltaH3[k] = sum * h3Out[k] * (1.0 - h3Out[k]);
                }

                // Hidden layer 2 delta
                for (size_t k = 0; k < hiddenSize; ++k) {
                    double sum = 0.0;
                    for (size_t j = 0; j < hiddenSize; ++j)
                        sum += W3[k + j * hiddenSize] * deltaH3[j];
                    deltaH2[k] = sum * h2Out[k] * (1.0 - h2Out[k]);
                }

                // Hidden layer 1 delta
                for (size_t k = 0; k < hiddenSize; ++k) {
                    double sum = 0.0;
                    for (size_t j = 0; j < hiddenSize; ++j)
                        sum += W2[k + j * hiddenSize] * deltaH2[j];
                    deltaH1[k] = sum * h1Out[k] * (1.0 - h1Out[k]);
                }

                // Update weights
                // W5 (hidden4->output)
                for (size_t j = 0; j < outputSize; ++j)
                    for (size_t k = 0; k < hiddenSize; ++k)
                        W5[k + j * hiddenSize] -= learningRate * deltaOut[j] * h4Out[k];

                // W4 (hidden3->hidden4)
                for (size_t j = 0; j < hiddenSize; ++j)
                    for (size_t k = 0; k < hiddenSize; ++k)
                        W4[k + j * hiddenSize] -= learningRate * deltaH4[j] * h3Out[k];

                // W3 (hidden2->hidden3)
                for (size_t j = 0; j < hiddenSize; ++j)
                    for (size_t k = 0; k < hiddenSize; ++k)
                        W3[k + j * hiddenSize] -= learningRate * deltaH3[j] * h2Out[k];

                // W2 (hidden1->hidden2)
                for (size_t j = 0; j < hiddenSize; ++j)
                    for (size_t k = 0; k < hiddenSize; ++k)
                        W2[k + j * hiddenSize] -= learningRate * deltaH2[j] * h1Out[k];

                // W1 (input->hidden1)
                for (size_t j = 0; j < hiddenSize; ++j)
                    for (size_t k = 0; k < inputSize; ++k)
                        W1[k + j * inputSize] -= learningRate * deltaH1[j] * input[k];
            }

            std::cout << "Epoch " << epoch << ", Total error: " << totalError << std::endl;
        }
    }

    std::vector<double> predict(const std::vector<double>& input) const {
        std::vector<double> h1(hiddenSize), h2(hiddenSize), h3(hiddenSize), h4(hiddenSize);
        return forward(input, h1, h2, h3, h4);
    }
};

int main() {
    std::vector<std::vector<double>> inputs = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 6.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 7.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 8.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 9.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 10.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 11.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.0, 2.0, 12.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
    };
    std::vector<int> outputs = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    NeuralNetwork nn(10, 4, 10, 0.01);  // 10 input features, 4 neurons per hidden layer, 10 output classes
    nn.train(inputs, outputs, 100);

    auto testOutput = nn.predict({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    std::cout << "\nTest outputs: ";
    for (double val : testOutput) std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}
