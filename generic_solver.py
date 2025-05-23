import csv
from collections import defaultdict
import pulp
from math import radians, sin, cos, sqrt, atan2

# Range in lat/long or radius (km) or max population
RANGE = 25_000_000
ACCOUNT_FOR_PREFIXES = True

CSV = "world50kcities.csv"

SOLVER = pulp.PULP_CBC_CMD
# SOLVER = pulp.GUROBI_CMD

def normalize(name):
    return name.replace("_", "").strip().lower()

def get_data():
    with open(CSV, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        cities = [{
            "lat": float(row[6]),
            "lon": float(row[7]),
            "population": int(row[4]),
            "names": [normalize(row[2]), normalize(row[3])]
        } for row in reader if row[10] == "a"]
    return cities

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378137
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def create_circle_name_coverage(city_list, name_to_city_indices):
    name_coverage = defaultdict(set)
    for name, indices_with_name in name_to_city_indices.items():
        for i in range(len(city_list)):
            city_i = city_list[i]
            for j in indices_with_name:
                city_j = city_list[j]
                distance = haversine_distance(city_i['lat'], city_i['lon'], city_j['lat'], city_j['lon'])
                if distance <= RANGE * 1000:
                    name_coverage[name].add(i)

    return name_coverage

def create_lat_long_coverage(city_list, name_to_city_indices, direction):
    merged_intervals = {}
    for name, indices in name_to_city_indices.items():
        # Create intervals for each city with this name
        intervals = []
        for idx in indices:
            d = city_list[idx][direction]
            intervals.append((d - RANGE, d + RANGE))
        
        # Merge overlapping intervals (same logic as original)
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (min(last[0], interval[0]), max(last[1], interval[1]))
                else:
                    merged.append(interval)
        merged_intervals[name] = merged

    # Determine which cities fall within each name's merged intervals
    name_coverage = defaultdict(set)
    for name, intervals in merged_intervals.items():
        for i, city in enumerate(city_list):
            d = city[direction]
            if any(start <= d <= end for (start, end) in intervals):
                name_coverage[name].add(i)
    
    return name_coverage

def create_pop_coverage(city_list, name_to_city_indices):
    name_coverage = defaultdict(set)
    for name, indices_with_name in name_to_city_indices.items():
        for j in indices_with_name:
            j_city = city_list[j]
            j_lat = j_city['lat']
            j_long = j_city['lon']
            sum_pop = j_city['population']
            coverage_set = {j}
            if sum_pop >= RANGE:
                name_coverage[name].update(coverage_set)
                continue
            
            # Collect other cities' distances and populations
            other_cities = []
            for i in range(len(city_list)):
                if i == j:
                    continue
                distance = haversine_distance(j_lat, j_long, city_list[i]['lat'], city_list[i]['lon'])
                other_cities.append((distance, i, city_list[i]['population']))
            
            # Sort by distance
            other_cities.sort(key=lambda x: x[0])
            
            # Accumulate populations until sum >= X
            for _, i, pop in other_cities:
                sum_pop += pop
                coverage_set.add(i)
                if sum_pop >= RANGE:
                    break
            
            name_coverage[name].update(coverage_set)
    
    return name_coverage


def build_prefix_map(names):
    """
    For every name find all (strict) prefixes that are themselves
    valid city names.
    """
    name_set = set(names)
    prefix_of = defaultdict(set)

    for n in names:
        for k in range(1, len(n)):
            p = n[:k]
            if p in name_set:
                prefix_of[n].add(p)
    return prefix_of

def main():
    city_list = get_data()
    print(f"Number of cities: {len(city_list)}")

    # Map each normalized name to all city indices where it appears
    name_to_city_indices = defaultdict(list)
    for i, city in enumerate(city_list):
        for name in city['names']:
            name_to_city_indices[name].append(i)

    # Determine coverage
    # name_coverage = create_circle_name_coverage(city_list, name_to_city_indices)
    # name_coverage = create_lat_long_coverage(city_list, name_to_city_indices, 'lon')
    name_coverage = create_pop_coverage(city_list, name_to_city_indices)

    # All names to consider
    names = list(name_coverage.keys())
    cost = {name: len(name) for name in names}
    print(f"Number of unique names considered: {len(names)}")
    
    # Set up ILP (Integer Linear Programming) problem
    prob = pulp.LpProblem("CityCover", pulp.LpMinimize)

    # Variables: 1 if name is selected, 0 otherwise
    vars = pulp.LpVariable.dicts("Name", names, cat='Binary')

    # Objective: minimize total character count
    prob += pulp.lpSum(vars[name] * cost[name] for name in names)

    # Constraints: every city must be covered by at least one selected name
    for i in range(len(city_list)):
        covering_names = [name for name in names if i in name_coverage[name]]
        if not covering_names:
            print(f"Error: City {i} (lat: {city_list[i]['lat']}) cannot be covered by any name.")
            return
        prob += pulp.lpSum(vars[name] for name in covering_names) >= 1

    # Constraint: ensure prefixes don't overlap
    # long_name chosen  ⇒  every prefix chosen
    if ACCOUNT_FOR_PREFIXES:
        print("Building prefixes...")
        prefix_of = build_prefix_map(names)
        for long, shorts in prefix_of.items():
            for sh in shorts:
                prob += vars[long] <= vars[sh] 

    # Solve the ILP
    print("Solving...")
    prob.solve(SOLVER(msg=False))

    # Check for optimality
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("No optimal solution found.")
        return

    # Output selected names
    selected = [name for name in names if pulp.value(vars[name]) >= 0.99]
    total_chars = sum(len(name) for name in selected)

    print(f"{pulp.LpStatus[prob.status].capitalize()} solution found.")
    print(f"Total characters used: {total_chars}")
    print(f"Number of names selected: {len(selected)}")
    print("Selected names:")
    print("\n".join(selected))

# Entry point
if __name__ == "__main__":
    main()