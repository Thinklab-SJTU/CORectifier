#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "CircleSector.h"
#include "Split.h"
using namespace std;


extern "C" {
	int* cvrp_local_search(
		short* tour,
		float *nodes_coords,
		float *demands,
		int nodes_num,
		int tour_length,
		int coords_scale,
		int demands_scale,
		int seed
	){
		// Problem Params
		Params params(nodes_coords, demands, nodes_num, coords_scale, demands_scale, seed);

		// The random initial policy may use more routes than the demand bound.
        int route_count = 0;
        for (int i = 1; i < tour_length; ++i)
            if (tour[i] == 0 && tour[i - 1] > 0) ++route_count;
        params.nbVehicles = std::max(params.nbVehicles, route_count);

        // Input Initial Tours
		Individual indiv(&params, tour, tour_length);
		Split split(&params);

		// Creating Local Search
		LocalSearch localSearch(&params, &split);

		// Local Search!
		localSearch.run(&indiv, params.penaltyCapacity, params.penaltyDuration);

		// Return
		int *ls_tours = new int[params.nbClients + params.nbVehicles + 2];
		if (indiv.isFeasible) {
			ls_tours[0] = 0;
			int ls_idx = 1;
			for (size_t i = 0; i < indiv.chromR.size(); ++i){
				if (!indiv.chromR[i].empty()) {
					for (int node : indiv.chromR[i]) {
						ls_tours[ls_idx] = node;
						ls_idx ++;
					}
					ls_tours[ls_idx] = 0;
					ls_idx ++;
				}
			}
			ls_tours[ls_idx] = -1;
		}
		else {
			ls_tours[0] = -1;
		}
		return ls_tours;
	}

    void free_cvrp_result(int* result) { delete[] result; }
}