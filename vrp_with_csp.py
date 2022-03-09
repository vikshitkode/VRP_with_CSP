# Solve the Capacited Vehicles Routing Problem (CVRP) 
# by combining iterative search with CSP
#
#
# The problem is described by specifying the coordinates of the points C,
# the demand of the points D, and the capacity of a vehicle Q.
# It is assumed that in C and D, the first point is the depot and the remaining
# points are the pickup points.
#
# Here we define as example a very small VRP with a depot at (0,0)
# and 4 pickup points, where the demand at every pickup point is 1 and
# the capacity of the vehicles is 2.
C = [[0,0], [1,0], [0, 2], [2, 0], [0, 1]]
D = [0] + [1] * (len(C) - 1)
Q = 2


# Google-OR tools is used for the CSP solver and the iterative search
# python -m pip install --upgrade --user ortools
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
from math import sqrt, ceil


def dist(a, b):
    """Euclidean distance between points a and b in the plane."""
    d = [a[0] - b[0], a[1] - b[1]]
    return sqrt(d[0] * d[0] + d[1] * d[1]) 


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [[dist(a, b) for b in C] for a in C]
    data['demands'] = D
    data['num_vehicles'] = ceil(sum(D) / Q)   
    data['vehicle_capacities'] = [Q] * data['num_vehicles']
    data['depot'] = 0
    return data


def initialization_by_csp(data):
    """Get a feasible solution to initialize the iterative search by solving 
    a CSP as described on pages 3--4 in the article 
    by Guimarans and Anton (2011)."""
    model = cp_model.CpModel()
    m = data['num_vehicles']
    n = len(data['distance_matrix']) - 1
    B = [[model.NewIntVar(0, 1, "b_" + str(i) + "_" + str(j)) 
          for j in range(n)] for i in range(m)]

    # Every pickup point is visited by exactly one vehicle
    for i in range(n):
        model.Add(sum([B[v][i] for v in range(m)]) == 1)

    # All vehicles must satisfy the capacity constraint
    for v in range(m):
        model.Add(cp_model.LinearExpr.ScalProd(B[v], data['demands'][1:]) <= Q)

    # Solve the CSP
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    v = [[solver.Value(B[i][j]) for j in range(n)] for i in range(m)]
    
    # Get the corresponding tours 
    # (We have to shift by 1 because the first point is the depot)
    return [[i + 1 for i, x in enumerate(v[j]) if x == 1] for j in range(m)]


def print_solution(data, manager, routing, solution):
    """Prints routing solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    
    data['initial_routes'] = initialization_by_csp(data)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    
    # Print the initial solution
    initial_solution = routing.ReadAssignmentFromRoutes(data['initial_routes'],
                                                        True)
    print('Initial solution:')
    print_solution(data, manager, routing, initial_solution)
    
    # Set default search parameters.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # Solve the problem.
    solution = routing.SolveFromAssignmentWithParameters(
        initial_solution, search_parameters)

    # Print improved solution
    print('\n\nSolution after search:')  
    print_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()
