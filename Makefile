CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra
INCLUDES = -I/usr/include/eigen3 -I/usr/include
LIBS = -lboost_system -lboost_filesystem -lboost_math_c99 -lcurl
TARGET = Static_Deep_MLP_4HiddenLayers
SOURCE = Static_Deep_MLP_4HiddenLayers.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f $(TARGET) *.o
