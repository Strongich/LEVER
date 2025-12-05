import copy
import math
from collections import deque

import networkx as nx

from policy_reusability.env.gridworld import GridWorld


class DAG:
    # n = node size
    # action size = number of possible actions
    # N = No. episodes
    # end node: the goal node
    # env length is only for gridworld
    def __init__(self, gridworld, N):
        gridworld.reset()
        self.gridworld = gridworld
        self.graph = nx.DiGraph()
        states = range(gridworld.state_count)
        self.graph.add_nodes_from(states)
        self.N = N
        self.end_node = gridworld.state_to_index(gridworld.target_position)
        self.action_size = gridworld.action_count
        self.env_length = gridworld.grid_length
        self.start_node = gridworld.state_to_index(gridworld.start_position)
        # Cache state/action lookups to reduce repeated conversions during graph ops
        self._state_cache = {
            idx: GridWorld.index_to_state(idx, self.env_length)
            for idx in self.graph.nodes
        }
        self._action_cache = {}
        # Static position sets for fast reward lookups
        self._gold_positions = (
            set(tuple(pos) for pos in getattr(gridworld, "gold_positions", []) or [])
        )
        self._block_positions = (
            set(tuple(pos) for pos in getattr(gridworld, "block_positions", []) or [])
        )
        self._target_position = tuple(getattr(gridworld, "target_position", ()))
        self._block_reward = getattr(gridworld, "block_reward", 0)
        self._target_reward = getattr(gridworld, "target_reward", 0)

    def add_edge(self, a, b):
        self.graph.add_edge(a, b)

    # This has been implemented for the gridworld environment with two actions: right and down

    def obtain_action(self, state_1_index, state_2_index):
        key = (state_1_index, state_2_index)
        if key in self._action_cache:
            return self._action_cache[key]

        state_1 = self._state_cache[state_1_index]
        state_2 = self._state_cache[state_2_index]

        # down
        if state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1]:
            action = 1
        # right
        elif state_2[0] == state_1[0] and state_2[1] == state_1[1] + 1:
            action = 0
        # down*2
        elif state_2[0] == state_1[0] + 2 and state_2[1] == state_1[1]:
            action = 3
        # right*2
        elif state_2[0] == state_1[0] and state_2[1] == state_1[1] + 2:
            action = 2
        # diagonal
        elif state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1] + 1:
            action = 4
        else:
            action = None
            print("Action could not be obtained")

        self._action_cache[key] = action
        return action

    # This has been implemented for the gridworld environment with two actions: right and down
    # def obtain_action(self, state_1_index, state_2_index):

    #     if (state_1_index == 0 and state_2_index == 2) or (state_1_index == 2 and state_2_index == 4) or (state_1_index == 3 and state_2_index == 5):
    #         return 1
    #     return 0

    # ENV width & length are only used for gridworld policy
    # to have a better understanding of the position of states

    def print(self, mode=0):
        # print(self.graph)
        if mode == 0:
            return
        elif mode == 1:
            n = self.graph.number_of_nodes()
            for i in range(n):
                print("node " + str(i) + ":")
                print("\t" + str(list(self.graph.neighbors(i))))
        elif mode == 2:
            print(self.graph.edges)
        elif mode == 3:
            n = self.graph.number_of_nodes()
            for i in range(n):
                print("node " + str(GridWorld.index_to_state(i, self.env_length)) + ":")
                neighbor_states = [
                    GridWorld.index_to_state(neighbor, self.env_length)
                    for neighbor in self.graph.neighbors(i)
                ]
                print("\t" + str(neighbor_states))

    # @staticmethod
    # def union_of_graphs(gridworld, graph_list):
    #
    #     union_graph = nx.DiGraph()
    #
    #     for graph in graph_list:
    #         union_graph.add_nodes_from(graph.nodes)
    #         union_graph.add_edges_from(graph.edges)
    #
    #     new_grid_world = copy.copy(gridworld)
    #     new_grid_world.reward_system = "combined"
    #     new_grid_world.reset()
    #     dag = DAG(new_grid_world, self.N)
    #     dag.graph = union_graph
    #     return dag

    @staticmethod
    def union_of_graphs(gridworld, dags, N):
        union_graph = nx.DiGraph()

        for dag in dags:
            union_graph.add_nodes_from(dag.graph.nodes)
            union_graph.add_edges_from(dag.graph.edges)

        new_grid_world = copy.copy(gridworld)
        new_grid_world.reward_system = "combined"
        new_grid_world.reset()
        dag = DAG(
            new_grid_world, N
        )  # Make sure you have self.N defined in the class or pass it as a parameter.
        dag.graph = union_graph

        return dag

    def union(self, other):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.graph.nodes)
        graph.add_edges_from(self.graph.edges)
        graph.add_edges_from(other.graph.edges)
        new_grid_world = copy.copy(self.gridworld)
        new_grid_world.reward_system = "combined"
        new_grid_world.reset()
        dag = DAG(new_grid_world, self.N)
        dag.graph = graph
        return dag

    def min_max_iter(self):
        graph_views = self._graph_views()
        return self.max_iter(graph_views=graph_views), self.min_iter(
            graph_views=graph_views
        )

    def _graph_views(self):
        topo_order = list(nx.topological_sort(self.graph))
        topo_position = {node: idx for idx, node in enumerate(topo_order)}
        predecessors = {
            node: list(self.graph.predecessors(node)) for node in self.graph.nodes
        }
        successors = {
            node: list(self.graph.successors(node)) for node in self.graph.nodes
        }
        return predecessors, successors, topo_position

    def max_iter(self, graph_views=None):
        predecessors, _, topo_position = graph_views or self._graph_views()
        visited = set()
        queue = deque([self.end_node])
        max_iterations = {node: [0] * self.action_size for node in self.graph.nodes}

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)
            preds = sorted(
                predecessors.get(next_node, ()),
                key=lambda n: topo_position[n],
                reverse=True,
            )

            in_degree_next = len(preds)
            next_values = max_iterations[next_node]
            next_total = sum(next_values)

            for node in preds:
                if node not in visited and node not in queue:
                    queue.append(node)

                action = self.obtain_action(node, next_node)
                if next_node == self.end_node:
                    max_iterations[node][action] = self.N - (in_degree_next - 1)
                else:
                    if (in_degree_next == 1) and (next_total > self.N):
                        max_iterations[node][action] = self.N
                    elif (in_degree_next > 1) and (next_total > self.N):
                        max_iterations[node][action] = self.N - (in_degree_next - 1)
                    else:
                        max_iterations[node][action] = next_total - (in_degree_next - 1)
        return max_iterations

    def calculate_itr_nodes(self, graph_views=None):
        _, successors, _ = graph_views or self._graph_views()
        itr = [0] * self.graph.number_of_nodes()
        in_degrees = self.graph.in_degree
        out_degrees = self.graph.out_degree

        for i in range(self.graph.number_of_nodes()):
            if i == self.start_node or i == self.end_node:
                itr[i] = self.N
                continue

            if not self._has_path_without_node(successors, i):
                itr[i] = self.N
                continue

            itr[i] = max(in_degrees(i), out_degrees(i))
        return itr

    def _has_path_without_node(self, successors, blocked_node):
        if self.start_node == blocked_node or self.end_node == blocked_node:
            return False

        visited = {blocked_node}
        queue = deque([self.start_node])

        while queue:
            node = queue.popleft()
            for nxt in successors.get(node, ()):
                if nxt in visited or nxt == blocked_node:
                    continue
                if nxt == self.end_node:
                    return True
                visited.add(nxt)
                queue.append(nxt)

        return False

    def min_iter(self, graph_views=None):
        predecessors, _, topo_position = graph_views or self._graph_views()
        visited = set()
        queue = deque([self.end_node])
        min_iterations = {node: [0] * self.action_size for node in self.graph.nodes}
        itr = self.calculate_itr_nodes(graph_views=graph_views)

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)
            for node in sorted(
                predecessors.get(next_node, ()),
                key=lambda n: topo_position[n],
                reverse=True,
            ):
                if node not in visited and node not in queue:
                    queue.append(node)
                action = self.obtain_action(node, next_node)
                if self.graph.out_degree(node) == 1 and self.graph.in_degree(node) > 1:
                    min_iterations[node][action] = itr[node]
                elif (
                    self.graph.in_degree(next_node) == 1
                    and self.graph.out_degree(next_node) > 1
                ):
                    min_iterations[node][action] = itr[next_node]
                else:
                    min_iterations[node][action] = 1
        return min_iterations

    def backtrack(
        self, min_iterations, max_iterations, learning_rate, discount_factor
    ):
        predecessors, _, topo_position = self._graph_views()
        visited = set()
        queue = deque([self.end_node])
        lower_Qs = {node: [0] * self.action_size for node in self.graph.nodes}
        upper_Qs = {node: [0] * self.action_size for node in self.graph.nodes}

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)

            for node in sorted(
                predecessors.get(next_node, ()),
                key=lambda n: topo_position[n],
                reverse=True,
            ):
                if node not in visited and node not in queue:
                    queue.append(node)

                action = self.obtain_action(node, next_node)
                min_iter, max_iter = (
                    min_iterations[node][action],
                    max_iterations[node][action],
                )

                # NOTE: update lower and upper bounds
                reward = self._calculate_reward_from_indices(node, next_node)
                next_max = max(
                    max_iterations[next_node][i] for i in range(self.action_size)
                )
                next_min = min(
                    min_iterations[next_node][i] for i in range(self.action_size)
                )
                upper_Qs[node][action] = round(
                    math.pow(-1, max_iter - 1)
                    * reward
                    * (
                        math.pow(learning_rate - 1, max_iter)
                        + math.pow(-1, max_iter - 1)
                    )
                    + (learning_rate * discount_factor * next_max),
                    2,
                )
                lower_Qs[node][action] = round(
                    math.pow(-1, min_iter - 1)
                    * reward
                    * (
                        math.pow(learning_rate - 1, min_iter)
                        + math.pow(-1, min_iter - 1)
                    )
                    + (learning_rate * discount_factor * next_min),
                    2,
                )
        return lower_Qs, upper_Qs

    def calculate_reward(self, state, next_state):
        self.gridworld.agent_position = next_state
        reward = self.gridworld._get_reward(state)
        return reward

    def _calculate_reward_from_indices(self, state_index, next_state_index):
        state = self._state_cache[state_index]
        next_state = self._state_cache[next_state_index]
        return self.calculate_reward(state, next_state)

    def _reward_path_static(self, current_state, next_state):
        if next_state in self._block_positions:
            return self._block_reward
        if next_state == self._target_position:
            return self._target_reward
        d1 = sum(abs(a - b) for a, b in zip(current_state, self._target_position))
        d2 = sum(abs(a - b) for a, b in zip(next_state, self._target_position))
        return d1 - d2

    def _reward_gold_static(self, next_state):
        if next_state in self._block_positions:
            return self._block_reward
        if next_state == self._target_position:
            return self._target_reward
        return 1 if next_state in self._gold_positions else 0

    def _edge_reward_static(self, state_index, next_state_index):
        # Replicates combined reward logic without mutating GridWorld state
        current_state = self._state_cache[state_index]
        next_state = self._state_cache[next_state_index]
        reward_system = getattr(self.gridworld, "reward_system", "path")

        if reward_system == "path":
            return self._reward_path_static(current_state, next_state)
        if reward_system == "gold":
            return self._reward_gold_static(next_state)
        if reward_system == "combined":
            return self._reward_gold_static(next_state) + self._reward_path_static(
                current_state, next_state
            )

        # Fallback to original (may mutate env) for unsupported reward systems
        return self._calculate_reward_from_indices(state_index, next_state_index)

    def compute_pruning_percentage(self, edge_count_before, edge_count_after):
        reduced_edge_count = edge_count_before - edge_count_after
        return round(((100 * reduced_edge_count) / edge_count_before), 2)

    def prune(self, lower_bounds, upper_bounds):
        queue = deque()
        queue.append(self.start_node)
        visited = set()
        edge_count_before = self.graph.number_of_edges()

        while queue:
            node = queue.popleft()
            visited.add(node)
            remove = set()
            next_nodes = list(self.graph.successors(node))
            if len(next_nodes) == 1:
                queue.append(next_nodes[0])
            else:
                bounds = {}
                for next_node in next_nodes:
                    action = self.obtain_action(node, next_node)
                    bounds[next_node] = (
                        lower_bounds[node][action],
                        upper_bounds[node][action],
                    )

                for next_node in next_nodes:
                    lower_bound, upper_bound = bounds[next_node]
                    for next_node_2 in next_nodes:
                        if (
                            (next_node == next_node_2)
                            or ((node, next_node) in remove)
                            or ((node, next_node_2) in remove)
                        ):
                            continue

                        lower_bound_2, upper_bound_2 = bounds[next_node_2]
                        if upper_bound_2 <= lower_bound:
                            remove.add((node, next_node_2))
                        else:
                            if (
                                next_node_2 not in queue
                                and next_node_2 not in visited
                            ):
                                queue.append(next_node_2)

                if remove:
                    self.graph.remove_edges_from(remove)
        edge_count_after = self.graph.number_of_edges()
        pruning_percentage = self.compute_pruning_percentage(
            edge_count_before=edge_count_before, edge_count_after=edge_count_after
        )
        return self.graph, pruning_percentage

    def find_paths(self):
        # Return generator to avoid materializing all paths (can be very large)
        return nx.all_simple_paths(
            self.graph, source=self.start_node, target=self.end_node
        )

    def best_path_dp(self):
        """
        Compute best path with dynamic programming over DAG edges (O(E) time/O(V) mem).
        Avoids enumerating all simple paths which is exponential for larger grids.
        """
        topological_order = list(nx.topological_sort(self.graph))
        dist = {node: float("-inf") for node in self.graph.nodes}
        parent = {}
        dist[self.start_node] = 0

        for node in topological_order:
            if dist[node] == float("-inf"):
                continue
            for succ in self.graph.successors(node):
                reward = self._edge_reward_static(node, succ)
                candidate = dist[node] + reward
                if candidate > dist[succ]:
                    dist[succ] = candidate
                    parent[succ] = node

        if dist[self.end_node] == float("-inf"):
            return None, float("-inf")

        # Reconstruct best path
        path = [self.end_node]
        cur = self.end_node
        while cur != self.start_node:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path, dist[self.end_node]
