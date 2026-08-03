from __future__ import annotations

import numpy as np
import heapq
import logging

from landlab.core.model_component import Component

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

logger = logging.getLogger("landslider")


class ShallowLandslideRunout(Component):
    """
    Subcomponent that routes shallow-landslide material downslope
    using flow-receiver topology and a distance-limited runout law.
    """

    _name = "ShallowLandslideRunout"

    _info = {
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "mapping": "node",
            "units": "m",
            "optional": False,
            "doc": "Soil thickness at nodes."
        },
        "hill_flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "mapping": "node",
            "units": "-",
            "optional": False,
            "doc": "Node IDs receiving routed hillslope flow from each node.",
        },
        "hill_flow__receiver_proportions": {
            "dtype": float,
            "intent": "in",
            "mapping": "node",
            "units": "-",
            "optional": False,
            "doc": "Proportions of hillslope flow routed to receiver nodes.",
        },
        "topographic__elevation": {
            "dtype": float,
            "units": "m",
            "intent": "in",
            "mapping": "node",
            "optional": False,
            "doc": "Land surface topographic elevation.",
        },
        "landslide__erosion": {
            "dtype": float,
            "units": "m",
            "intent": "out",
            "mapping": "node",
            "optional": False,
            "doc": "Soil thickness removed from each node by the latest runout step.",
        },
        "landslide__deposition": {
            "dtype": float,
            "units": "m",
            "intent": "out",
            "mapping": "node",
            "optional": False,
            "doc": "Soil thickness deposited at each node by the latest runout step.",
        },
        "landslide__soil_depth_change": {
            "dtype": float,
            "units": "m",
            "intent": "out",
            "mapping": "node",
            "optional": False,
            "doc": "Net soil-depth change (deposition minus erosion) from runout.",
        },
    }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __init__(self, grid):
        super().__init__(grid)
        self._last_failed_nodes = np.array([], dtype=int)
        self._last_paths = []
        self._last_path_proportions = np.array([], dtype=float)
        self._last_path_details = {}
        self._last_source_proportion_sums = {}
        self._last_source_path_counts = {}
        self._last_erosion = np.zeros(grid.number_of_nodes, dtype=float)
        self._last_deposition = np.zeros(grid.number_of_nodes, dtype=float)
        
        required = (
            "hill_flow__receiver_node",
            "hill_flow__receiver_proportions",
        )

        missing = [f for f in required if f not in grid.at_node]

        if missing:
            raise RuntimeError(
                "ShallowLandslideRunout requires flow routing fields, "
                f"but missing: {missing}. "
                "Run a FlowAccumulator or router before enabling runout."
            )

        receivers = np.asarray(grid.at_node["hill_flow__receiver_node"])
        proportions = np.asarray(grid.at_node["hill_flow__receiver_proportions"])
        if receivers.ndim != 2 or proportions.ndim != 2:
            raise ValueError(
                "ShallowLandslideRunout requires multiflow hill routing so sediment "
                "can be divided among receiver paths. Configure "
                "PriorityFloodFlowRouter with separate_hill_flow=True and a "
                "multiple-flow hill_flow_metric such as 'Quinn'."
            )

        for name, metadata in self._info.items():
            if metadata["intent"] == "out" and name not in grid.at_node:
                grid.add_zeros(name, at="node", dtype=metadata["dtype"])

    @property
    def results(self):
        """Return diagnostics owned by the latest multiflow runout step."""
        return {
            "failed_nodes": self._last_failed_nodes,
            "paths": self._last_paths,
            "path_proportions": self._last_path_proportions,
            "path_details": self._last_path_details,
            "source_proportion_sums": self._last_source_proportion_sums,
            "source_path_counts": self._last_source_path_counts,
            "erosion": self._last_erosion,
            "deposition": self._last_deposition,
            "soil_depth_change": self._grid.at_node[
                "landslide__soil_depth_change"
            ],
        }


    def run_one_step(
        self,
        failed_nodes: np.ndarray,
        runout_distance: np.ndarray,
        return_paths: bool = False,
    ):
        """
        Route failed shallow-landslide material downslope and update
        soil thickness on the grid.
        """

        erosion_field = self._grid.at_node["landslide__erosion"]
        deposition_field = self._grid.at_node["landslide__deposition"]
        soil_change_field = self._grid.at_node["landslide__soil_depth_change"]
        erosion_field.fill(0.0)
        deposition_field.fill(0.0)
        soil_change_field.fill(0.0)
        self._last_failed_nodes = np.asarray(failed_nodes, dtype=int).copy()
        self._last_paths = []
        self._last_path_proportions = np.array([], dtype=float)
        self._last_path_details = {}
        self._last_source_proportion_sums = {}
        self._last_source_path_counts = {}

        if failed_nodes.size == 0:
            self._last_erosion = erosion_field.copy()
            self._last_deposition = deposition_field.copy()
            return None

        paths, proportions, details = self._trace_paths_landslides(
            failed_nodes, runout_distance
        )
        self._last_paths = paths
        self._last_path_proportions = np.asarray(proportions, dtype=float)
        self._last_path_details = details
        for path, proportion in zip(paths, proportions):
            if not path or proportion <= 0:
                continue
            source = int(path[0])
            self._last_source_proportion_sums[source] = (
                self._last_source_proportion_sums.get(source, 0.0)
                + float(proportion)
            )
            self._last_source_path_counts[source] = (
                self._last_source_path_counts.get(source, 0) + 1
            )

        soil, erosion, deposition = self._update_soil_depth(
            paths,
            proportions,
            self._grid.at_node["soil__depth"],
        )

        self._grid.at_node["soil__depth"][:] = soil

        erosion_field[:] = erosion
        deposition_field[:] = deposition
        soil_change_field[:] = deposition - erosion

        # cache diagnostics (optional but useful)
        self._last_erosion = erosion
        self._last_deposition = deposition

        if return_paths:
            return details

    # %% Helpers
    def _calculate_node_distance(self, node1, node2):
        grid = self._grid
        x1, y1 = grid.node_x[node1], grid.node_y[node1]
        x2, y2 = grid.node_x[node2], grid.node_y[node2]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _trace_paths_landslides(
        self,
        starting_nodes,
        newmark_distances,
        ):
        grid = self._grid

        receiver_nodes = grid.at_node["hill_flow__receiver_node"]
        receiver_props = grid.at_node["hill_flow__receiver_proportions"]
        boundary_nodes = set(grid.boundary_nodes)

        final_paths = []
        final_proportions = []
        path_details = {}
        
        logger.info("Tracing landslide runout paths...")

        for node in starting_nodes:
            max_dist = newmark_distances[node]
            stack = [(0.0, node, 1.0, [node])]
            path_details[node] = []

            while stack:
                dist, current, prop, path = heapq.heappop(stack)

                if dist >= max_dist:
                    final_paths.append(tuple(path))
                    final_proportions.append(prop)
                    path_details[node].append((path, prop))
                    continue

                recs = receiver_nodes[current]
                props = receiver_props[current]
                positive = props > 0
                valid_receivers = positive & (recs >= 0) & (recs != current)
                stopped_proportion = float(
                    np.sum(props[positive & ((recs < 0) | (recs == current))])
                )

                # Quinn can assign a non-zero share to the current node. Keep
                # that share as a terminated branch rather than silently
                # renormalizing it into the moving receivers.
                if stopped_proportion > 0:
                    stopped_weight = prop * stopped_proportion
                    final_paths.append(tuple(path))
                    final_proportions.append(stopped_weight)
                    path_details[node].append((path, stopped_weight))

                # Pit or outlet cell.
                if not np.any(valid_receivers):
                    if stopped_proportion <= 0:
                        final_paths.append(tuple(path))
                        final_proportions.append(prop)
                        path_details[node].append((path, prop))
                    continue

                for r, p in zip(recs[valid_receivers], props[valid_receivers]):

                    new_dist = dist + self._calculate_node_distance(current, r)
                    new_prop = prop * p
                    new_path = path + [r]

                    if (
                        r in boundary_nodes
                        or np.isnan(grid.at_node["topographic__elevation"][r])
                    ):
                        # This receiver branch terminates at the current node.
                        # Retain its own multiflow weight; using the parent's
                        # weight here would duplicate mass when several
                        # receiver branches meet the boundary.
                        final_paths.append(tuple(path))
                        final_proportions.append(new_prop)
                        path_details[node].append((path, new_prop))
                    else:
                        heapq.heappush(
                            stack, (new_dist, r, new_prop, new_path)
                        )

        return final_paths, final_proportions, path_details
    
    def _update_soil_depth(
        self,
        paths,
        proportions,
        soil_depth,
        ):
        # Restore the 2025 source-to-endpoint transport concept: receiver
        # topology determines each branch and its proportion, while soil is
        # eroded only at the initiating node and deposited only at endpoints.
        # Every branch uses the source's original depth. Normalizing the
        # terminated branch weights removes the historical order-dependent
        # under-excavation (e.g., sequential 0.5/0.5 moved only 75%).
        soil = soil_depth.copy()
        erosion = np.zeros_like(soil)
        deposition = np.zeros_like(soil)

        paths_by_source = {}
        for path, prop in zip(paths, proportions):
            if not path or prop <= 0:
                continue
            paths_by_source.setdefault(int(path[0]), []).append(
                (int(path[-1]), float(prop), len(path) > 1)
            )

        logger.info("Updating soil depth...")
        for src, endpoints in paths_by_source.items():
            available = float(soil_depth[src])
            total_weight = sum(weight for _, weight, _ in endpoints)
            has_moving_path = any(moved for _, _, moved in endpoints)
            if available <= 0 or total_weight <= 0 or not has_moving_path:
                continue

            remaining = available
            for index, (dst, weight, _) in enumerate(endpoints):
                moved = (
                    remaining
                    if index == len(endpoints) - 1
                    else available * weight / total_weight
                )
                moved = min(max(moved, 0.0), remaining)
                remaining -= moved
                deposition[dst] += moved
            erosion[src] = available - remaining

        soil -= erosion
        soil += deposition

        return soil, erosion, deposition

