CXX ?= c++
CXXFLAGS ?= -O3 -std=c++11 -fPIC
SOURCES = cvrp_local_search.cpp Params.cpp LocalSearch.cpp Individual.cpp Split.cpp
HEADERS = $(wildcard *.h)

cvrp_local_search.so: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -shared $(SOURCES) -o $@

.PHONY: clean
clean:
	rm -f cvrp_local_search.so
