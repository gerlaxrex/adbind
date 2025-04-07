CXX = g++
CXXFLAGS = -o make/
CXXVERSION = -std=c++17

$(shell mkdir -p make)

.PHONY: clean

main: Variable.so main.o
	$(CXX) -g make/main.o make/Variable.so $(CXXVERSION) $(CXXFLAGS)main

main.o: main.cpp Variable.hpp
	$(CXX) -c main.cpp $(CXXVERSION) $(CXXFLAGS)main.o

Variable.so: Variable.cpp Variable.hpp
	$(CXX) -fPIC Variable.cpp $(CXXVERSION) -shared -Wall $(CXXFLAGS)Variable.so

# Variable.o: Variable.cpp Variable.hpp
# 	$(CXX) -c Variable.cpp $(CXXVERSION) $(CXXFLAGS)Variable.o

clean:
	rm -r make