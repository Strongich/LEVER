import networkx as nx
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value
from DAG import DAG

def create_dag():
    start = "START"
    end = "END"
    G = nx.DiGraph()
    G.add_nodes_from([start, end, "C", "B", "D", "E"])  
    G.add_edges_from([(start, "C"), (start, "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E"), ("D", end), ("E", end)])  
    return G, start, end

def compute_itr(G, N, start, end):
    itr = {node: 0 for node in G.nodes()}

    for i in G.nodes():
        graph_copy = G.copy()
        graph_copy.remove_node(i)
        if i == start or i == end:
            itr[i] = N
        elif not nx.has_path(graph_copy, source=start, target=end):
            itr[i] = N
        else:
            itr[i] = max(G.in_degree(i), G.out_degree(i))
    return itr

def f(G, node):
    a = G.out_degree(node)
    b = G.in_degree(node)
    if a == 1 and b > 1:
        return True, (node, list(G.successors(node))[0])
    elif a > 1 and b == 1:
        list(G.successors(node))[0]
        return True, (list(G.predecessors(node))[0], node)
    else:
        return False, None

def build_constraints(G, itr, N, end, M, E):
    constraints = []

    variables = {}
    for edge in G.edges:
        variables[edge] = (LpVariable(name=f"{edge[0]}_{edge[1]}_l", lowBound=0, cat='Integer'),
                           LpVariable(name=f"{edge[0]}_{edge[1]}_u", lowBound=0, cat='Integer'))

    objective_function = lpSum(variables[edge][1] - variables[edge][0] for edge in G.edges)

    for node in G.nodes():
        cond, t = f(G, node)
        if cond:
            constraints.append(variables[t][0] == itr[node])

    for edge in G.edges:
        constraints.append(variables[edge][0] <= variables[edge][1])  
        constraints.append(variables[edge][0] <= N)  
        constraints.append(variables[edge][0] >= 1)  
        constraints.append(variables[edge][1] <= N)  
        constraints.append(variables[edge][1] >= 1)  
        #print(constraints)

        s = edge[0]
        s_prime = edge[1]
        p = G.in_degree(s_prime)
        out_deg_s = G.out_degree(s)
        in_deg_s_prime = G.in_degree(s_prime)
        

        if out_deg_s > 1 and in_deg_s_prime > 1:
            constraints.append(variables[edge][0] == 1)

        
        # Define constraint for total
        total = lpSum(variables[(s_prime, next_node)][1] for next_node in G.successors(s_prime))
        
        total_exceeds_N = LpVariable(name=f"{s}_{s_prime}_exceeds_N", cat='Binary')
        # Add constraints to ensure that exceeds_N is set to 1 if total > N, and 0 otherwise
        constraints.append(total - (N + E) <= M * total_exceeds_N)
        constraints.append((N + E) - total <= (M * (1 - total_exceeds_N)))

        if s_prime == end:
            constraints.append(variables[edge][1] == N - (p - 1))
        else:
            indegree_equal_zero = p == 0
            indegree_equal_one = p == 1
            indegree_greater_one = p > 1

            #If total <= N:
             #   ti_u = total – (p – 1)
            constraints.append(variables[edge][1] - (total - (p - 1)) - M * (total_exceeds_N) <= 0)
            constraints.append(variables[edge][1] - (total - (p - 1)) + M * (total_exceeds_N) >= 0)

            # total > N and If indegree(next_node) == 0:
            # ti_u= total – (p – 1)
            constraints.append(variables[edge][1] - (total - (p - 1)) - M * (1 - total_exceeds_N) - M * (1 - indegree_equal_zero) <= 0)
            constraints.append(variables[edge][1] - (total - (p - 1)) + M * (1 - total_exceeds_N) + M * (1 - indegree_equal_zero) >= 0)

            # total > N and If indegree(next_node) == 1:
            # ti_u= N
            constraints.append(variables[edge][1] - (N) - M * (1 - total_exceeds_N) - M * (1 - indegree_equal_one) <= 0)
            constraints.append(variables[edge][1] - (N) + M * (1 - total_exceeds_N) + M * (1 - indegree_equal_one) >= 0)

            # total > N and If indegree(next_node) > 1:
            # ti_u= N – (p – 1)
            constraints.append(variables[edge][1] - (N - (p - 1)) - M * (1 - total_exceeds_N) - M * (1 - indegree_greater_one) <= 0)
            constraints.append(variables[edge][1] - (N - (p - 1)) + M * (1 - total_exceeds_N) + M * (1 - indegree_greater_one) >= 0)

        # if s_prime == "END":
        #     constraints.append(variables[edge][1] == N - (p - 1))
        # else:
        #     total = sum((variables[(s_prime, next_node)][1]) for next_node in G.successors(s_prime))
        #     if G.in_degree(s_prime) == 1 and total > N:  
        #         constraints.append(variables[edge][1] == N)
        #     elif G.in_degree(s_prime) > 1 and total > N:  
        #         constraints.append(variables[edge][1] == N - (p - 1)) 
        #     else:
        #         constraints.append(variables[edge][1] == total - (p - 1))

    return objective_function, constraints, variables

def solve_integer_programming(dag, objective_function, constraints, variables, get_dag=True):
    prob = LpProblem("Integer_Programming_Problem", LpMinimize)
    prob += objective_function

    for constraint in constraints:
        prob += constraint
    prob.solve()

    print("Objective Function Value: ", value(prob.objective))
    for constraint in constraints:
        print(constraint, "=", value(constraint))
    print("*************")
    for var_name, (var_lower, var_upper) in variables.items():
        print(f"{var_name}_l =", value(var_lower))
        #print(f"{var_name}_u =", value(var_upper))
    
    for var_name, (var_lower, var_upper) in variables.items():
        #print(f"{var_name}_l =", value(var_lower))
        print(f"{var_name}_u =", value(var_upper))

    if get_dag:
        min_iters = {}
        max_iters = {}
        for var_name, (var_lower, var_upper) in variables.items():
            state_tuple = var_lower.name[:-2].split("_")
            s1, s2 = int(state_tuple[0]), int(state_tuple[1])
            action = dag.obtain_action(s1, s2)
            min_iters[(s1, action)] = value(var_lower)
            max_iters[(s1, action)] = value(var_upper)
        min_iterations = []
        max_iterations = []
        for i in range(dag.gridworld.state_count):
            min_iterations.append([])
            max_iterations.append([])
            min_iterations = [[0] * dag.gridworld.action_count for _ in range(dag.gridworld.state_count)]
            max_iterations = [[0] * dag.gridworld.action_count for _ in range(dag.gridworld.state_count)]

                
        for i in range(dag.gridworld.state_count):
            for j in range(dag.gridworld.action_count):
                if (i, j) in min_iters:
                    min_iterations[i][j] = min_iters[(i, j)]
                if (i, j) in max_iters:
                    max_iterations[i][j] = max_iters[(i, j)]
        return (min_iterations, max_iterations)
    else:
        min_iters = []
        max_iters = []
        for var_name, (var_lower, var_upper) in variables.items():
            min_iters.append(value(var_lower))
            max_iters.append(value(var_upper))
        return (min_iters, max_iters)

def solve_IP(dag, N, start, end, M=100000, E=0.00001, get_dag=True):
    if get_dag:
        itr = compute_itr(dag.graph, N, start, end)
        objective_function, constraints, variables = build_constraints(dag.graph, itr, N, end, M, E)
        return solve_integer_programming(dag, objective_function, constraints, variables)
    else:
        itr = compute_itr(dag, N, start, end)
        objective_function, constraints, variables = build_constraints(dag, itr, N, end, M, E)
        return solve_integer_programming(dag, objective_function, constraints, variables, get_dag)

def main():
    M = 100000
    E = 0.00001
    N = 4
    G, start, end = create_dag()
    
    min_Itrs, max_Itrs = solve_IP(G, N, start, end, M, E, get_dag=False)
    print(min_Itrs)
    print(max_Itrs)

if __name__ == "__main__":
    main()
