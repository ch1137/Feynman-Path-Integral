import numpy as np

#Define the number of spacial latice points, N
N = 7
#Define particle mass, m
m = 1.0
#Define complex time interval, T
T = 4
#Define starting point, x0
x0 = -2.0
#Set the well type
well_type = 'harmonic oscillator'
#Set number of iterations for each path
iterations = 1000
#Set number of paths to simulate
num_paths = 5

dt = T/N


#Define different well types
def harmonic_oscillator(x):
    return 0.5 * x**2

#Well types dictionary
potential = {'harmonic oscillator': harmonic_oscillator}


def path_generator(x0, N):
    #Function to generate a path of N points starting from x0.
    #Parameters:    x0 (float), N (int)
    #Returns:       path (array): A 1D array containing the generated path.
    path = np.zeros(N)
    path[0] = x0
    for i in range(N-1):
        path[i+1] = path[i] + np.random.uniform(-1, 1)
    return path

def sum_energy(Path):
    #Function to calculate the sum of energy along a given path.
    #Parameters:    Path (list)
    #Returns:       Total energy (float): The total energy along the path.
    e = 0
    E = 0
    for i in range(len(Path)-1):
        e = ((m/2) * ((Path[i+1] - Path[i])/dt)**2 
            + potential[well_type]((Path[i+1] + Path[i])/2))
        E += e
    return E

def path_acceptor(old_path, new_path):
    #Function that runs through criteria to accept or reject perturbed paths.
    #Parameters:    old_path (list), new_path (list)
    #Returns:       accept (bool): True if new path should be accepted.
    accept = False
    E_change = sum_energy(new_path) - sum_energy(old_path)
    if E_change < 0:
        accept = True
    elif np.random.uniform(0, 1) < np.exp(-dt * E_change):
        accept = True
    return accept

def point_perturber(Path, point):
    #Function to perturb a random point in the path using gaussian random noise.
    #Parameters:    Path (list)
    #Returns:       new path (array): A 1D array containing the perturbed path.
    Path[point] += np.random.normal(0, 0.05)
    return Path

def path_perturber(Path):
    #Function to perturb the entire path using gaussian random noise.
    #Parameters:    Path (list)
    #Returns:       new path (array): A 1D array containing the perturbed path.
    for point in range(1, len(Path)):
        new_path = point_perturber(Path, point)
        if path_acceptor(Path, new_path):
            Path = new_path.copy()
    return Path

def path_simulator(x0, N, iterations):
    #Function to generate and iterate pertibations of one path and calculate the total energy.
    #Parameters:    x0 (float), N (int), iterations (int)
    #Returns:       path (array): A 1D array containing the final path after iterations and total energy.
    path = path_generator(x0, N)
    for i in range(iterations):
        path = path_perturber(path)
    return path

def appending_energy(Path):
    #Function to append the total energy of a path to the end of the path.
    #Parameters:    Path (list)
    #Returns:       path (array): A 1D array containing the path with total energy appended.
    path = np.append(Path, sum_energy(Path))
    return path

def multi_path_simulator(x0, N, iterations, num_paths):
    #Function to generate and iterate pertibations of multiple paths.
    #Parameters:    x0 (float), N (int), iterations (int), num_paths (int)
    #Returns:       paths (array): A 2D array containing the final paths after iterations.
    paths = np.zeros((num_paths, N+1))
    for i in range(num_paths):
        paths[i,:] = appending_energy(path_simulator(x0, N, iterations))
    return paths

def path_averager(Paths):
    #Function to calculate the weighted average path.
    #Parameters:    Paths (2D array)
    #Returns:       average_path (array): A 1D array containing the weighted average path.
    weightings = np.exp(-dt * Paths[:,-1])
    Total_weight = np.sum(weightings)
    for i in range(num_paths):
        Paths[i,:] *= weightings[i]
    average_path = np.sum(Paths, axis=0)
    return average_path[0:-1]/Total_weight

def run_paths():
    #Function to run the entire path simulation and averaging process.
    #Returns:       average_path (array): A 1D array containing the weighted average path.
    Paths = multi_path_simulator(x0, N, iterations, num_paths)
    average_path = path_averager(Paths)
    return average_path