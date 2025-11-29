#include <iostream>
#include <cmath>      // For std::exp, std::log, std::sqrt
#include <random>     // For std::mt19937, std::normal_distribution
#include <algorithm>  // For std::max, std::transform, std::max_element, std::distance, std::copy_n, std::fill
#include <iomanip>    // For std::setw, std::fixed, std::setprecision
#include <array>      // For std::array
#include <chrono>     // For std::chrono::high_resolution_clock
#include <functional> // For std::plus, std::minus, std::multiplies
#include <stdexcept>  // For std::out_of_range

constexpr size_t INPUT_SIZE = 10;
constexpr size_t HIDDEN_SIZE = 4;
constexpr size_t OUTPUT_SIZE = 10;
constexpr double LEARNING_RATE = 0.01;
constexpr int EPOCHS = 100;

// --- StaticVector ---
// A fixed-size vector wrapper around std::array for small, compile-time sized vectors.
template<size_t N>
struct StaticVector {
    std::array<double, N> data;

    // Default constructor zero-initializes `data` for arithmetic types (C++11 standard and later).
    StaticVector() = default;

    // Allow initializer list construction
    StaticVector(std::initializer_list<double> list) {
        if (list.size() > N) {
            throw std::out_of_range("Initializer list size exceeds StaticVector capacity");
        }
        // std::array elements are value-initialized (zero-initialized for double)
        // when StaticVector is constructed. So, elements beyond list.size() are already 0.0.
        std::copy_n(list.begin(), list.size(), data.begin());
    }

    // Element access operators
    // Noexcept is safe assuming correct index usage by caller in optimized builds.
    double& operator[](size_t i) noexcept { return data[i]; }
    const double& operator[](size_t i) const noexcept { return data[i]; }

    // Iterators
    typename std::array<double, N>::iterator begin() noexcept { return data.begin(); }
    typename std::array<double, N>::const_iterator begin() const noexcept { return data.begin(); }
    typename std::array<double, N>::iterator end() noexcept { return data.end(); }
    typename std::array<double, N>::const_iterator end() const noexcept { return data.end(); }
    typename std::array<double, N>::const_iterator cbegin() const noexcept { return data.cbegin(); }
    typename std::array<double, N>::const_iterator cend() const noexcept { return data.cend(); }

    // Size query
    constexpr size_t size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return N == 0; }

    // Compound assignment operators using std::transform for potential vectorization
    StaticVector<N>& operator+=(const StaticVector<N>& other) noexcept {
        std::transform(data.begin(), data.end(), other.data.begin(), data.begin(), std::plus<double>());
        return *this;
    }
    StaticVector<N>& operator-=(const StaticVector<N>& other) noexcept {
        std::transform(data.begin(), data.end(), other.data.begin(), data.begin(), std::minus<double>());
        return *this;
    }
    StaticVector<N>& operator*=(double scalar) noexcept {
        std::transform(data.begin(), data.end(), data.begin(), [scalar](double val){ return val * scalar; });
        return *this;
    }
    StaticVector<N>& operator/=(double scalar) noexcept {
        std::transform(data.begin(), data.end(), data.begin(), [scalar](double val){ return val / scalar; });
        return *this;
    }
    // Element-wise multiplication (Hadamard product)
    StaticVector<N>& operator*=(const StaticVector<N>& other) noexcept {
        std::transform(data.begin(), data.end(), other.data.begin(), data.begin(), std::multiplies<double>());
        return *this;
    }
};

// --- Global StaticVector Operators ---
// Pass lhs by value for +=, -=, *= (scalar) to use copy-and-swap idiom, which is efficient.
// For element-wise multiplication, also pass by value to leverage `operator*=`.
template<size_t N>
[[nodiscard]] StaticVector<N> operator+(StaticVector<N> lhs, const StaticVector<N>& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template<size_t N>
[[nodiscard]] StaticVector<N> operator-(StaticVector<N> lhs, const StaticVector<N>& rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

template<size_t N>
[[nodiscard]] StaticVector<N> operator*(StaticVector<N> lhs, double scalar) noexcept {
    lhs *= scalar;
    return lhs;
}

template<size_t N>
[[nodiscard]] StaticVector<N> operator*(double scalar, StaticVector<N> rhs) noexcept {
    rhs *= scalar;
    return rhs;
}

// Element-wise multiplication (Hadamard product)
template<size_t N>
[[nodiscard]] StaticVector<N> operator*(StaticVector<N> lhs, const StaticVector<N>& rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

// --- StaticMatrix ---
// A fixed-size matrix wrapper around std::array for small, compile-time sized matrices.
// Stores data in row-major order.
template<size_t Rows, size_t Cols>
struct StaticMatrix {
    std::array<double, Rows * Cols> data;

    // Default constructor zero-initializes `data` for arithmetic types
    StaticMatrix() = default;

    constexpr size_t rows() const noexcept { return Rows; }
    constexpr size_t cols() const noexcept { return Cols; }

    // Matrix-Vector Multiplication: result = M * vec
    [[nodiscard]] StaticVector<Rows> operator*(const StaticVector<Cols>& vec) const noexcept {
        StaticVector<Rows> result{}; // Zero-initialized by default constructor
        if constexpr (Rows > 0 && Cols > 0) {
            for (size_t r = 0; r < Rows; ++r) {
                double sum = 0.0;
                // This loop structure is cache-friendly for row-major `data` access and `vec` access.
                for (size_t c = 0; c < Cols; ++c) {
                    sum += data[r * Cols + c] * vec[c];
                }
                result[r] = sum;
            }
        }
        return result;
    }

    // Transpose-Vector Multiplication: result = M^T * vec
    // Optimized for better cache locality when accessing matrix data.
    [[nodiscard]] StaticVector<Cols> transpose_multiply(const StaticVector<Rows>& vec) const noexcept {
        StaticVector<Cols> result{}; // Zero-initialized
        if constexpr (Rows > 0 && Cols > 0) {
            // Iterate over rows of M (and elements of vec)
            for (size_t r = 0; r < Rows; ++r) {
                const double vec_r_val = vec[r]; // Cache vec[r] value
                // Iterate over columns of M, updating respective result elements.
                // This provides contiguous access to data[r * Cols + c] for fixed 'r'.
                for (size_t c = 0; c < Cols; ++c) {
                    result[c] += data[r * Cols + c] * vec_r_val; // Access (r, c) of M
                }
            }
        }
        return result;
    }

    // Element access operators
    double& operator()(size_t r, size_t c) noexcept {
        return data[r * Cols + c];
    }
    const double& operator()(size_t r, size_t c) const noexcept {
        return data[r * Cols + c];
    }
};

// --- Activation Functions ---

template<size_t N>
void softmax_activation(StaticVector<N>& vec) noexcept {
    if (vec.empty()) return;

    // Find max value for numerical stability (prevents overflow/underflow)
    double max_val = *std::max_element(vec.cbegin(), vec.cend());

    // Compute exponentials and sum in a single pass using std::transform
    double sum_exp = 0.0;
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [&](double val) {
                       double exp_val = std::exp(val - max_val);
                       sum_exp += exp_val;
                       return exp_val;
                   });

    constexpr double SOFTMAX_DENOMINATOR_EPSILON = 1e-12; // Epsilon for stability
    if (sum_exp > SOFTMAX_DENOMINATOR_EPSILON) {
        std::transform(vec.begin(), vec.end(), vec.begin(),
                       [sum_exp](double val) { return val / sum_exp; });
    } else {
        // Fallback for extremely small sums (e.g., all inputs were very negative).
        // Set to uniform distribution to avoid NaNs.
        double uniform_val = 1.0 / static_cast<double>(N);
        std::fill(vec.begin(), vec.end(), uniform_val);
    }
}

inline double relu_activation(double x) noexcept {
    return std::max(0.0, x);
}

inline double relu_derivative(double x) noexcept {
    return (x > 0.0) ? 1.0 : 0.0; // Derivative is 0 for x <= 0, 1 for x > 0
}

template<size_t N>
[[nodiscard]] StaticVector<N> vector_relu_derivative(const StaticVector<N>& z_values) noexcept {
    StaticVector<N> derivatives{};
    std::transform(z_values.cbegin(), z_values.cend(), derivatives.begin(), relu_derivative);
    return derivatives;
}

// --- Neural Network Class ---
class NeuralNetwork {
private:
    const double m_learningRate;

    // Weights
    StaticMatrix<HIDDEN_SIZE, INPUT_SIZE> m_W1;
    StaticMatrix<HIDDEN_SIZE, HIDDEN_SIZE> m_W2;
    StaticMatrix<HIDDEN_SIZE, HIDDEN_SIZE> m_W3;
    StaticMatrix<HIDDEN_SIZE, HIDDEN_SIZE> m_W4;
    StaticMatrix<OUTPUT_SIZE, HIDDEN_SIZE> m_W5;

    // Biases (default-initialized to zeros by StaticVector constructor)
    StaticVector<HIDDEN_SIZE> m_b1;
    StaticVector<HIDDEN_SIZE> m_b2;
    StaticVector<HIDDEN_SIZE> m_b3;
    StaticVector<HIDDEN_SIZE> m_b4;
    StaticVector<OUTPUT_SIZE> m_b5;

    // Mutable members for storing intermediate forward/backward pass values.
    mutable StaticVector<HIDDEN_SIZE> m_hidden1Z;
    mutable StaticVector<HIDDEN_SIZE> m_hidden1Out;
    mutable StaticVector<HIDDEN_SIZE> m_hidden2Z;
    mutable StaticVector<HIDDEN_SIZE> m_hidden2Out;
    mutable StaticVector<HIDDEN_SIZE> m_hidden3Z;
    mutable StaticVector<HIDDEN_SIZE> m_hidden3Out;
    mutable StaticVector<HIDDEN_SIZE> m_hidden4Z;
    mutable StaticVector<HIDDEN_SIZE> m_hidden4Out;
    mutable StaticVector<OUTPUT_SIZE> m_outputZ;
    mutable StaticVector<OUTPUT_SIZE> m_finalOutput;

    // Mutable members for backpropagation deltas
    mutable StaticVector<OUTPUT_SIZE> m_deltaO;
    mutable StaticVector<HIDDEN_SIZE> m_deltaH4;
    mutable StaticVector<HIDDEN_SIZE> m_deltaH3;
    mutable StaticVector<HIDDEN_SIZE> m_deltaH2;
    mutable StaticVector<HIDDEN_SIZE> m_deltaH1;

    std::mt19937 m_generator;

    // Helper to initialize weights using He initialization (for ReLU)
    template<size_t R, size_t C>
    void initialize_weights(StaticMatrix<R, C>& weights, double std_dev) {
        std::normal_distribution<double> distribution(0.0, std_dev);
        // Direct iteration over the underlying std::array for simplicity and performance.
        for (double& weight : weights.data) {
            weight = distribution(m_generator);
        }
    }

public:
    NeuralNetwork(double rate)
        : m_learningRate(rate),
          m_generator(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
    {
        // He initialization: std_dev = sqrt(2 / fan_in) for ReLU
        double std_dev_W1 = std::sqrt(2.0 / static_cast<double>(INPUT_SIZE));
        double std_dev_hidden = std::sqrt(2.0 / static_cast<double>(HIDDEN_SIZE));
        double std_dev_W5 = std::sqrt(2.0 / static_cast<double>(HIDDEN_SIZE)); // Fan-in from last hidden layer

        initialize_weights(m_W1, std_dev_W1);
        initialize_weights(m_W2, std_dev_hidden);
        initialize_weights(m_W3, std_dev_hidden);
        initialize_weights(m_W4, std_dev_hidden);
        initialize_weights(m_W5, std_dev_W5);
    }

    // Forward pass: computes outputs for a given input
    [[nodiscard]] const StaticVector<OUTPUT_SIZE>& forward(const StaticVector<INPUT_SIZE>& input_vec) const noexcept {
        // Layer 1
        m_hidden1Z = m_W1 * input_vec + m_b1;
        std::transform(m_hidden1Z.cbegin(), m_hidden1Z.cend(), m_hidden1Out.begin(), relu_activation);

        // Layer 2
        m_hidden2Z = m_W2 * m_hidden1Out + m_b2;
        std::transform(m_hidden2Z.cbegin(), m_hidden2Z.cend(), m_hidden2Out.begin(), relu_activation);

        // Layer 3
        m_hidden3Z = m_W3 * m_hidden2Out + m_b3;
        std::transform(m_hidden3Z.cbegin(), m_hidden3Z.cend(), m_hidden3Out.begin(), relu_activation);

        // Layer 4
        m_hidden4Z = m_W4 * m_hidden3Out + m_b4;
        std::transform(m_hidden4Z.cbegin(), m_hidden4Z.cend(), m_hidden4Out.begin(), relu_activation);

        // Output Layer
        m_outputZ = m_W5 * m_hidden4Out + m_b5;

        // Apply Softmax activation (in-place on m_finalOutput)
        m_finalOutput = m_outputZ; // Copy m_outputZ to m_finalOutput before activation
        softmax_activation(m_finalOutput);

        return m_finalOutput;
    }

    // Training function: performs a forward pass, backpropagation, and weight/bias updates
    void train(const StaticVector<INPUT_SIZE>& input_vec, const StaticVector<OUTPUT_SIZE>& target_vec) noexcept {
        (void)forward(input_vec); // Populate intermediate values (Z, Out)

        // Calculate output layer delta (for Softmax + Cross-Entropy: output - target)
        m_deltaO = m_finalOutput - target_vec;

        // Backpropagate deltas through hidden layers
        // delta_h = (W_next^T * delta_next) * ReLU_derivative(z_h) (element-wise multiplication)
        m_deltaH4 = (m_W5.transpose_multiply(m_deltaO)) * vector_relu_derivative(m_hidden4Z);
        m_deltaH3 = (m_W4.transpose_multiply(m_deltaH4)) * vector_relu_derivative(m_hidden3Z);
        m_deltaH2 = (m_W3.transpose_multiply(m_deltaH3)) * vector_relu_derivative(m_hidden2Z);
        m_deltaH1 = (m_W2.transpose_multiply(m_deltaH2)) * vector_relu_derivative(m_hidden1Z);

        const double lr_scalar = m_learningRate;

        // Update weights and biases using SGD
        // W_new = W_old - learning_rate * delta * output_prev (outer product implicitly)
        // b_new = b_old - learning_rate * delta

        // Output layer (W5, b5)
        for (size_t k = 0; k < OUTPUT_SIZE; ++k) {
            const double delta_k = m_deltaO[k];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                m_W5(k, j) -= lr_scalar * delta_k * m_hidden4Out[j];
            }
            m_b5[k] -= lr_scalar * delta_k;
        }

        // Hidden layer 4 (W4, b4)
        for (size_t k = 0; k < HIDDEN_SIZE; ++k) {
            const double delta_k = m_deltaH4[k];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                m_W4(k, j) -= lr_scalar * delta_k * m_hidden3Out[j];
            }
            m_b4[k] -= lr_scalar * delta_k;
        }

        // Hidden layer 3 (W3, b3)
        for (size_t k = 0; k < HIDDEN_SIZE; ++k) {
            const double delta_k = m_deltaH3[k];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                m_W3(k, j) -= lr_scalar * delta_k * m_hidden2Out[j];
            }
            m_b3[k] -= lr_scalar * delta_k;
        }

        // Hidden layer 2 (W2, b2)
        for (size_t k = 0; k < HIDDEN_SIZE; ++k) {
            const double delta_k = m_deltaH2[k];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                m_W2(k, j) -= lr_scalar * delta_k * m_hidden1Out[j];
            }
            m_b2[k] -= lr_scalar * delta_k;
        }

        // Hidden layer 1 (W1, b1)
        for (size_t k = 0; k < HIDDEN_SIZE; ++k) {
            const double delta_k = m_deltaH1[k];
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                m_W1(k, j) -= lr_scalar * delta_k * input_vec[j];
            }
            m_b1[k] -= lr_scalar * delta_k;
        }
    }

    // Prediction function (simply calls forward pass)
    [[nodiscard]] const StaticVector<OUTPUT_SIZE>& predict(const StaticVector<INPUT_SIZE>& input_vec) const noexcept {
        return forward(input_vec);
    }
};

// --- Main Function ---
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Example input data
    StaticVector<INPUT_SIZE> input_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    size_t label = 1; // Target class index

    // Create one-hot encoded target vector
    StaticVector<OUTPUT_SIZE> target_data{}; // All elements initialized to 0.0
    if (label < OUTPUT_SIZE) {
        target_data[label] = 1.0;
    } else {
        std::cerr << "Warning: Provided label (" << label << ") is out of OUTPUT_SIZE range. Using default zero target.\n";
    }

    NeuralNetwork nn(LEARNING_RATE); // Instantiate neural network

    std::cout << "Starting Training (" << EPOCHS << " Epochs)....\n";
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        nn.train(input_data, target_data); // Perform one training step

        // Periodically print loss for monitoring
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            const StaticVector<OUTPUT_SIZE>& current_output = nn.predict(input_data);
            double totalLoss = 0.0;
            constexpr double CROSS_ENTROPY_LOG_EPSILON = 1e-9; // Small epsilon for log stability

            // For one-hot encoded targets, loss is simply -log(P_target_class).
            if (label < OUTPUT_SIZE) {
                 totalLoss = -std::log(current_output[label] + CROSS_ENTROPY_LOG_EPSILON);
            } else {
                // Fallback for an invalid label (should ideally not happen with proper data validation)
                for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
                    if (target_data[i] > 0.0) {
                        totalLoss += -target_data[i] * std::log(current_output[i] + CROSS_ENTROPY_LOG_EPSILON);
                    }
                }
            }

            std::cout << "Epoch " << std::setw(3) << epoch + 1 << " Loss (Cross-Entropy): "
                      << std::fixed << std::setprecision(6) << totalLoss << '\n';
        }
    }
    std::cout << "Training Finished.\n";

    // Make a final prediction
    const StaticVector<OUTPUT_SIZE>& prediction = nn.predict(input_data);

    // Print prediction results
    std::cout << "\n--- Prediction for Input ---\n";
    std::cout << "Output Vector (Probabilities): [";
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < prediction.size(); ++i) {
        std::cout << prediction[i] << (i < prediction.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // Find the predicted class (index of maximum probability)
    size_t max_idx = static_cast<size_t>(std::distance(prediction.cbegin(),
                                                       std::max_element(prediction.cbegin(), prediction.cend())));
    std::cout << "Predicted Class (Index): " << max_idx << '\n';
    std::cout << "Actual Class: " << label << '\n';

    return 0;
}
