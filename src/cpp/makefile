CXX = g++
CXXFLAGS = -o make/
CXXVERSION = -std=c++17
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
EXT_SUFFIX = $(shell python3-config --extension-suffix)

$(shell mkdir -p make)

.PHONY: clean

# since adbind is not linked we need to insert in all  
all: main adbind$(EXT_SUFFIX)

main: Variable.so main.o
	$(CXX) -g make/main.o make/Variable.so $(CXXVERSION) $(CXXFLAGS)main

main.o: main.cpp Variable.hpp
	$(CXX) -c main.cpp $(CXXVERSION) $(CXXFLAGS)main.o

Variable.so: Variable.cpp Variable.hpp
	$(CXX) -O3 -Wall -shared $(CXXVERSION) -fPIC Variable.cpp $(CXXFLAGS)Variable.so

adbind$(EXT_SUFFIX): Variable.cpp Variable.hpp | make
	$(CXX) -O3 -Wall -shared $(CXXFLAGS) -fPIC $(PYBIND_INCLUDES) Variable.cpp $(CXXFLAGS)adbind$(EXT_SUFFIX)

clean:
	rm -r make