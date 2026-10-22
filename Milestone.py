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

dt = T/N


#Define different well types
def harmonic_oscillator(x):
    return 0.5 * x**2

#Well types dictionary
potential = {'harmonic oscillatotor': harmonic_oscillator}


def path_generator(x0, N):
    #Function to generate a path of N points starting from x0.
    #Parameters:    x0 (float), N (int)
    #Returns:       path(array): A 1D array containing the generated path.
    path = np.zeros(N)
    path[0] = x0
    for i in range(N-1):
        path[i+1] = path[i] + np.random.uniform(-1, 1)
    return path

def path_perturber(Path):
    #Function to perturb a given path by adding random noise.
    #Parameters:    Path (list)
    #Returns:       new path (array): A 1D array containing the perturbed path.
    for i in range(1, len(Path)):
        Path[i] = Path[i] + np.random.uniform(-0.1, 0.1)
    return Path


def sum_energy(Path):
    #Function to calculate the sum of energy along a given path.
    #Parameters:    Path (list)
    #Returns:       Total energy (float): The total energy along the path.
    e = 0
    E = 0
    for i in range(len(Path)-1):
        e = ((m/2) * ((Path[i+1] - Path[i])/dt)**2 
            - potential[well_type]((Path[i+1] + Path[i])/2))
        E += e
    return E

