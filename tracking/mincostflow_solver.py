import cvxpy as cp

def solve_cell_tracking_flow(M, N, capacities, costs):
    """
    Solve a min-cost flow problem for cell tracking with splits but no merges.
    
    Args:
        M (int): Number of Left nodes (L, corresponding to detections at frame t).
        N (int): Number of Right nodes (R, corresponding to detections at frame t+1).
        capacities (dict): Dictionary mapping (u, v) edge to maximum allowed flow.
        costs (dict): Dictionary mapping (u, v) edge to cost value.
    
    Returns:
        dict: Dictionary of flow variables { (u, v): optimal flow value }.
    """

    edges = costs.keys()

    # Define flow variables for each edge
    flow = { (u, v): cp.Variable(integer=True) for (u, v) in edges }

    # Objective: Minimize total cost
    cost_expr = cp.sum([flow[(u, v)] * costs[(u, v)] for (u, v) in edges])

    # Constraints list
    constraints = []

    # Capacity constraints: 0 <= flow <= capacity
    for (u, v) in edges:
        constraints.append(flow[(u, v)] >= 0)
        constraints.append(flow[(u, v)] <= capacities[(u, v)])

    # Source T+ node constraint: total outgoing flow
    constraints.append(
        cp.sum([flow[("T+", f"L{i}")] for i in range(1, M+1)]) +
        flow[("T+", "A")] == M + N
    )

    # Target T- node constraint: total incoming flow
    constraints.append(
        cp.sum([flow[(f"R{j}", "T-")] for j in range(1, N+1)]) +
        flow[("D", "T-")] == M + N
    )

    # Flow conservation at Left nodes (L_i): outgoing flow equals 1
    for i in range(1, M+1):
        constraints.append(
            flow[("T+", f"L{i}")] ==
            cp.sum([flow[(f"L{i}", f"R{j}")] for j in range(1, N+1) if (f"L{i}", f"R{j}") in edges]) +
            flow[(f"L{i}", "D")] +
            cp.sum([flow[(f"L{i}", f"S_{i}_{j}_{k}")] for j in range(1, N+1) for k in range(j+1, N+1) if (f"L{i}", f"S_{i}_{j}_{k}") in edges])
        )

    # Flow conservation at Right nodes (R_j)
    for j in range(1, N+1):
        constraints.append(
            cp.sum([
                flow[(f"L{i}", f"R{j}")] for i in range(1, M+1) if (f"L{i}", f"R{j}") in edges
            ]) +
            flow[("A", f"R{j}")] +
            cp.sum([
                flow[(f"S_{i}_{j}_{k}", f"R{j}")] for i in range(1, M+1) for k in range(j+1, N+1) if (f"S_{i}_{j}_{k}", f"R{j}") in edges
            ]) +
            cp.sum([
                flow[(f"S_{i}_{l}_{j}", f"R{j}")] for i in range(1, M+1) for l in range(1, j) if (f"S_{i}_{l}_{j}", f"R{j}") in edges
            ])
            == flow[(f"R{j}", "T-")]
        )

    # Flow conservation at Appear node (A)
    constraints.append(
        flow[("T+", "A")] ==
        cp.sum([flow[("A", f"R{j}")] for j in range(1, N+1)]) +
        cp.sum([
            flow[("A", f"S_{i}_{j}_{k}")]
            for i in range(1, M+1) for j in range(1, N+1) for k in range(j+1, N+1)
            if ("A", f"S_{i}_{j}_{k}") in edges
        ]) +
        flow[("A", "D")]
    )

    # Flow conservation at Disappear node (D)
    constraints.append(
        flow[("D", "T-")] ==
        cp.sum([flow[(f"L{i}", "D")] for i in range(1, M+1)]) +
        flow[("A", "D")]
    )

    # Split nodes constraints: flow in = flow out
    for i in range(1, M+1):
        for j in range(1, N+1):
            for k in range(j+1, N+1):
                if (f"L{i}", f"S_{i}_{j}_{k}") in edges and (f"S_{i}_{j}_{k}", f"R{j}") in edges and (f"S_{i}_{j}_{k}", f"R{k}") in edges:
                    constraints.append(
                        flow[(f"L{i}", f"S_{i}_{j}_{k}")] + flow[("A", f"S_{i}_{j}_{k}")] ==
                        flow[(f"S_{i}_{j}_{k}", f"R{j}")] + flow[(f"S_{i}_{j}_{k}", f"R{k}")]
                    )
                    # Ensure equality between all parts (split must be symmetric)
                    constraints.append(flow[(f"L{i}", f"S_{i}_{j}_{k}")] == flow[("A", f"S_{i}_{j}_{k}")])
                    constraints.append(flow[(f"L{i}", f"S_{i}_{j}_{k}")] == flow[(f"S_{i}_{j}_{k}", f"R{j}")])
                    constraints.append(flow[(f"L{i}", f"S_{i}_{j}_{k}")] == flow[(f"S_{i}_{j}_{k}", f"R{k}")])

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost_expr), constraints)
    problem.solve(solver=cp.GUROBI)  # You can replace GUROBI with another solver if needed

    if problem.status == cp.OPTIMAL:
        return flow
    else:
        print("No feasible solution found.")
        return None
