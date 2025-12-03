import networkx as nx
from collections import deque
from env.gridworld import GridWorld
import math
import copy


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

    def add_edge(self, a, b):
        self.graph.add_edge(a, b)

    # This has been implemented for the gridworld environment with two actions: right and down

    def obtain_action(self, state_1_index, state_2_index):

        state_1 = GridWorld.index_to_state(state_1_index, self.env_length)
        state_2 = GridWorld.index_to_state(state_2_index, self.env_length)

        # down
        if (state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1]):
            return 1
        # right
        elif (state_2[0] == state_1[0] and state_2[1] == state_1[1] + 1):
            return 0
        # down*2
        if (state_2[0] == state_1[0] + 2 and state_2[1] == state_1[1]):
            return 3
        # right*2
        elif (state_2[0] == state_1[0] and state_2[1] == state_1[1] + 2):
            return 2
        #diagonal
        elif (state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1] + 1):
            return 4
        else:
            print("Action could not be obtained")

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
                neighbor_states = [GridWorld.index_to_state(neighbor, self.env_length) for neighbor in self.graph.neighbors(i)]
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
        dag = DAG(new_grid_world, N)  # Make sure you have self.N defined in the class or pass it as a parameter.
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
        return self.max_iter(), self.min_iter()


    def max_iter(self):
        visited = set()
        queue = deque([self.end_node])
        max_iterations = {node: [0] * self.action_size for node in self.graph.nodes}

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)
            adding_candidates = []
            
            for node in self.graph.predecessors(next_node):
                if node not in visited and node not in queue:
                    adding_candidates.append(node)
                action = self.obtain_action(node, next_node)
                if next_node == self.end_node:
                    max_iterations[node][action] = self.N - (self.graph.in_degree(next_node) - 1)
                else:
                    total = sum(max_iterations[next_node][i] for i in range(self.action_size))
                    if (self.graph.in_degree(next_node) == 1) and (total > self.N):
                        max_iterations[node][action] = self.N
                    elif (self.graph.in_degree(next_node) > 1) and (total > self.N):
                        max_iterations[node][action] = self.N - (self.graph.in_degree(next_node) - 1)
                    else:
                        max_iterations[node][action] = total - (self.graph.in_degree(next_node) - 1)
                #this is where we should add the nodes from adding candidates to the queue 
                for i in range(len(adding_candidates)):
                    for j in range(i + 1, len(adding_candidates)):
                        if (self.graph.has_edge(adding_candidates[i], adding_candidates[j])):
                            adding_candidates[i], adding_candidates[j] = adding_candidates[j], adding_candidates[i]
                queue.extend(adding_candidates)
        return max_iterations


    def calculate_itr_nodes(self):
        itr = [0] * self.graph.number_of_nodes()

        for i in range(self.graph.number_of_nodes()):
            graph_copy = self.graph.copy()
            graph_copy.remove_node(i)
            # if node is disconnected from the whole graph, simply ignore it
            if i == self.start_node or i == self.end_node:
                itr[i] = self.N
            elif not nx.has_path(graph_copy, source=self.start_node, target=self.end_node):
                itr[i] = self.N
            else:
                itr[i] = max(self.graph.in_degree(i), self.graph.out_degree(i))
        return itr


    def min_iter(self):
        visited = set()
        queue = deque([self.end_node])
        min_iterations = {node: [0] * self.action_size for node in self.graph.nodes}
        itr = self.calculate_itr_nodes()

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)
            for node in self.graph.predecessors(next_node):
                if node not in visited and node not in queue:
                    queue.append(node)
                action = self.obtain_action(node, next_node)
                if self.graph.out_degree(node) == 1 and self.graph.in_degree(node) > 1:
                    min_iterations[node][action] = itr[node]
                elif self.graph.in_degree(next_node) == 1 and self.graph.out_degree(next_node) > 1:
                    min_iterations[node][action] = itr[next_node]
                else:
                    min_iterations[node][action] = 1
        return min_iterations


    def backtrack(self, min_iterations, max_iterations, learning_rate, discount_factor):
        visited = set()
        queue = deque([self.end_node])
        lower_Qs = {node: [0] * self.action_size for node in self.graph.nodes}
        upper_Qs = {node: [0] * self.action_size for node in self.graph.nodes}

        while queue:
            next_node = queue.popleft()
            visited.add(next_node)
            adding_candidates = []

            for node in self.graph.predecessors(next_node):
                if node not in visited and node not in queue:
                    adding_candidates.append(node)

                action = self.obtain_action(node, next_node)
                min_iter, max_iter = min_iterations[node][action], max_iterations[node][action]

                # NOTE: update lower and uppder bounds
                reward = self.calculate_reward(self.gridworld.index_to_state(node, self.env_length), self.gridworld.index_to_state(next_node, self.env_length))
                # Assuming max_iters is a dictionary with nodes as keys and lists as values
                next_max = max(max_iterations[next_node][i] for i in range(self.action_size))
                next_min = min(min_iterations[next_node][i] for i in range(self.action_size))
                upper_Qs[node][action] = round(math.pow(-1, max_iter - 1) * reward * (math.pow(learning_rate - 1, max_iter) + math.pow(-1, max_iter - 1)) + (learning_rate * discount_factor * next_max), 2)
                lower_Qs[node][action] = round(math.pow(-1, min_iter - 1) * reward * (math.pow(learning_rate - 1, min_iter) + math.pow(-1, min_iter - 1)) + (learning_rate * discount_factor * next_min), 2)

                #this is where we should add the nodes from adding candidates 
                for i in range(len(adding_candidates)):
                    for j in range(i + 1, len(adding_candidates)):
                        if (self.graph.has_edge(adding_candidates[i], adding_candidates[j])):
                            adding_candidates[i], adding_candidates[j] = adding_candidates[j], adding_candidates[i]
                queue.extend(adding_candidates)
        return lower_Qs, upper_Qs


    def calculate_reward(self, state, next_state):
        self.gridworld.agent_position = next_state
        reward = self.gridworld._get_reward(state)
        return reward


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
            remove = []
            next_nodes = list(self.graph.successors(node))
            if len(next_nodes) == 1:
                queue.append(next_nodes[0])
            else:
                for next_node in next_nodes:
                    action = self.obtain_action(node, next_node)
                    lower_bound = lower_bounds[node][action]
                    upper_bound = upper_bounds[node][action]
                    for next_node_2 in next_nodes:
                        if (next_node == next_node_2) or ((node, next_node) in remove) or ((node, next_node_2) in remove):
                            continue
                        else:
                            action_2 = self.obtain_action(node, next_node_2)
                            upper_bound_2 = upper_bounds[node][action_2]
                            lower_bound_2 = lower_bounds[node][action_2]

                            if upper_bound_2 <= lower_bound:
                                # print("Edge removed: " + str((node, next_node_2)))
                                remove.append((node, next_node_2))
                            else:
                                if next_node_2 not in queue and next_node_2 not in visited:
                                    queue.append(next_node_2)
                    # if remove != []:
                    #     print("removed edges:", str(remove))

                    self.graph.remove_edges_from(remove)
        edge_count_after = self.graph.number_of_edges()
        pruning_percentage = self.compute_pruning_percentage(edge_count_before=edge_count_before, edge_count_after=edge_count_after)
        return self.graph, pruning_percentage


    def find_paths(self):
        paths = []
        #x_start, y_start = self.gridworld.index_to_state(self.start_node, self.gridworld.grid_length)
        #x_end, y_end = self.gridworld.index_to_state(self.end_node, self.gridworld.grid_length)
        for path in nx.all_simple_paths(self.graph, source= self.start_node, target = self.end_node):
            paths.append(path)
        return paths
                

            
