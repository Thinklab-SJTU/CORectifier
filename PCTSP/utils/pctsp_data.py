r"""
Basic solver for Prize Collection Traveling Salesman Problem (PCTSP).

In the PCTSP, each node has an associated prize and penalty. Unlike the traditional TSP, where
all cities must be visited, the PCTSP allows for skipping some nodes. The final objective is
to minimize the sum of the traveled distance and the sum of the penalties for unvisited nodes
while satisfying a minimum total prize constraint.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np

def iterative_execution_for_file(file, message="", show_time=False):
    return file


class PCTSPSolver:
    """Read the original labeled PCTSP text format."""
    def __init__(self, precision=np.float32):
        self.precision = precision

    def from_data(self, depots, points, penalties, norm_prizes, tours,
                  ref=False, norm="EUC_2D", normalize=False):
        if normalize or norm != "EUC_2D":
            raise ValueError("Released PCTSP data use normalized Euclidean coordinates.")
        self.depots = np.asarray(depots, dtype=self.precision)
        self.points = np.asarray(points, dtype=self.precision)
        self.penalties = np.asarray(penalties, dtype=self.precision)
        self.norm_prizes = np.asarray(norm_prizes, dtype=self.precision)
        if ref:
            self.ref_tours = tours
        else:
            self.tours = tours

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        r"""
        Read data from `.txt` file.

        :param file_path: string, path to the `.txt` file containing PCTSP instances data.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver.
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param show_time: boolean, whether the data is being read with a visual progress display.

        .. dropdown:: Example

            ::

                >>> from ml4co_kit import PCTSPSolver

                # create PCTSPSolver
                >>> solver = PCTSPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/pctsp/txt/pctsp50_concorde.txt")
                >>> solver.points.shape
                (16, 50, 2)
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            # record to lists
            depots_list = list()
            points_list = list()
            penalties_list = list()
            norm_prizes_list = list()
            tours_list = list()

            # read by lines
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                # line to strings
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depots = split_line_1[0]
                split_line_2 = split_line_1[1].split(" penalties ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" norm_prizes ")
                penalties = split_line_3[0]
                split_line_4 = split_line_3[1].split(" output ")
                norm_prizes = split_line_4[0]
                tours = split_line_4[1]

                # strings to array
                depots = depots.split(" ")
                depots = np.array([float(depots[0]), float(depots[1])], dtype=self.precision)
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                penalties = penalties.split(" ")
                penalties = np.array([
                    float(penalties[i]) for i in range(len(penalties))
                ], dtype=self.precision)
                norm_prizes = norm_prizes.split(" ")
                norm_prizes = np.array(
                    [float(norm_prizes[i]) for i in range(len(norm_prizes))], dtype=self.precision
                )
                tours = tours.split(" ")
                tours = np.array(
                    [int(tours[i]) for i in range(len(tours))]
                )

                # add to the list
                depots_list.append(depots)
                points_list.append(points)
                penalties_list.append(penalties)
                norm_prizes_list.append(norm_prizes)
                tours_list.append(tours)

        # check if return list
        if return_list:
            return (
                depots_list, points_list, penalties_list, \
                norm_prizes_list, tours_list
            )

        depots = np.array(depots_list)
        points = np.array(points_list)
        penalties = np.array(penalties_list)
        norm_prizes = np.array(norm_prizes_list)
        tours = tours_list

        # use ``from_data``
        self.from_data(
            depots=depots, points=points, penalties=penalties,
            norm_prizes=norm_prizes, tours=tours, ref=ref,
            norm=norm, normalize=normalize
        )
