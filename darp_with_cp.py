DATA_DIRECTORY = 'data' # we assume that this directory has all the benchmark datafiles (and nothing but these files)
OUTPUTFILE = 'output10A.csv' # results will be written here
RUNTIME = 10 # Runtime in seconds for each algorithm

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
from math import sqrt, ceil
import csv
import os

def dist(a, b):
    """Euclidean distance between points a and b in the plane."""
    d = [a[0] - b[0], a[1] - b[1]]
    return sqrt(d[0] * d[0] + d[1] * d[1]) 

def create_data_model(datafile):
    """Stores the data for the problem."""
    with open(datafile) as f:
        rows = csv.reader(f, delimiter=' ', skipinitialspace=True)
        r = next(rows)
        n = int(r[1])
        C = []
        for r in rows:
            C.append([float(r[1]), float(r[2])])
        del C[-1] # coordinates of depot are repeated in the last line; we remove them here.
    data = {}
    data['distance_matrix'] = [[dist(a, b) for b in C] for a in C]
    data['pickups_deliveries'] = [[i, n + i] for i in range(1, n + 1)]
    data['num_vehicles'] = ceil(n / 2)
    data['max_dist'] = ceil(3 * max([max(x) for x in data['distance_matrix']]))
    data['depot'] = 0
    return data

def initialization_by_csp(data):
    """Get an feasible solution to initialize the iterative search by solving 
    a CSP as described on pages 3--4 in the article 
    by Guimarans and Anton (2011)"""
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
        model.Add(sum(B[v]) <= ceil(n / m))

    # Solve the CSP
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    v = [[solver.Value(B[i][j]) for j in range(n)] for i in range(m)]
    
    # Get the corresponding tours 
    # (We have to shift by 1 because the first point is the depot)
    return [[i + 1 for i, x in enumerate(v[j]) if x == 1] for j in range(m)]


def get_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        total_distance += route_distance
    return total_distance


def DARP(datafile):
    """Solve the DARP in datafile first with VNS+CS and then with basic VNS."""
    # Instantiate the data problem.
    data = create_data_model(datafile)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc.
    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        data['max_dist'],  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Define Transportation Requests.
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))

    # Solve with CSP
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = RUNTIME
    initial_solution = routing.ReadAssignmentFromRoutes(initialization_by_csp(data), True)
    solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
    if solution:
        s1 = get_solution(data, manager, routing, solution)
    else:
        s1 = -1 

    # Solve without CSP
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    search_parameters.time_limit.seconds = RUNTIME
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        s2 = get_solution(data, manager, routing, solution)
    else:
        s2 = -1
        
    return [datafile, s1, s2]

# Loop over all the files in datafiles in DATA_DIRECTORY;
# Solve the DARP problems in these files with function DARP;
# And finally write all the results into OUTPUTFILE
with open(OUTPUTFILE, 'w', newline='', encoding='utf-8') as outputfile:
    w = csv.writer(outputfile)
    for f in os.listdir(DATA_DIRECTORY):
        d = os.path.join(DATA_DIRECTORY, f)
        if os.path.isfile(d):
            # print(d)
            r = DARP(d)
            w.writerow(r)
        