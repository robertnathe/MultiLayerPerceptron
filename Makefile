CXX := g++
CXXFLAGS := -Wall -Wextra -O3 -march=native -std=c++17
TARGET := output
SRC := MultiLayerPerceptron.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
