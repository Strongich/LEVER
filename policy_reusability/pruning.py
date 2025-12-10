import time

from policy_reusability.integer_programming import solve_IP


def compute_pruning(before, after):
    reduced_edge_count = before - after
    return (100 * reduced_edge_count) / before


def get_best_path(gridworld, dag, paths):
    best_path = None
    max_reward = 0
    for path in paths:
        reward = 0
        gridworld.reset()
        for i in range(len(path) - 1):
            state_index_1 = path[i]
            state_index_2 = path[i + 1]
            action = dag.obtain_action(state_index_1, state_index_2)
            grid, r, done, _ = gridworld.step(action)
            reward += r
        if reward >= max_reward:
            max_reward = reward
            best_path = path
    return best_path, max_reward


def run_pruning(gridworld, dag_1, dag_2, learning_rate, discount_factor):
    start_time = time.time()
    union_dag = dag_1.union(dag_2)
    # print("Union DAG:")
    union_dag.print()
    max_iterations, min_iterations = union_dag.min_max_iter()
    print("Min iterations: \n", min_iterations)
    print("Max iterations: \n", max_iterations)
    lower_bounds, upper_bounds = union_dag.backtrack(
        min_iterations, max_iterations, learning_rate, discount_factor
    )
    print("Run backtrack!")

    edge_count_before_prune = union_dag.graph.number_of_edges()
    pruned_graph, pruning_percentage = union_dag.prune(lower_bounds, upper_bounds)

    total_time = time.time() - start_time
    best_path, max_reward = union_dag.best_path_dp()
    return best_path, max_reward, total_time, pruning_percentage


def run_pruning_IP(gridworld, dag_1, dag_2, learning_rate, discount_factor, N):
    start_time = time.time()
    union_dag = dag_1.union(dag_2)
    # print("Union DAG:")
    union_dag.print()
    max_iterations, min_iterations = solve_IP(
        union_dag, N, union_dag.start_node, union_dag.end_node
    )
    print("Min iterations: \n", min_iterations)
    print("Max iterations: \n", max_iterations)
    lower_bounds, upper_bounds = union_dag.backtrack(
        min_iterations, max_iterations, learning_rate, discount_factor
    )

    # edge_count_before_prune = union_dag.graph.number_of_edges()
    pruned_graph, pruning_percentage = union_dag.prune(lower_bounds, upper_bounds)

    total_time = time.time() - start_time
    best_path, max_reward = union_dag.best_path_dp()
    return best_path, max_reward, total_time, pruning_percentage
