CXX = g++
CXXFLAGS = -o make/
CXXVERSION = -std=c++17

$(shell mkdir -p make)

main: Variable.o main.o
	$(CXX) -g make/main.o make/Variable.o $(CXXVERSION) $(CXXFLAGS)main

main.o: main.cpp Variable.hpp
	$(CXX) -c main.cpp $(CXXVERSION) $(CXXFLAGS)main.o

Variable.o: Variable.cpp Variable.hpp
	$(CXX) -c Variable.cpp $(CXXVERSION) $(CXXFLAGS)Variable.o