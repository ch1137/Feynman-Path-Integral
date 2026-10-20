#Define the number of spacial latice points, N
N = 7
#Define particle mass, m
m = 1.0
#Define complex time interval, T
T = 4

dt = T/N

def potential(x):
    return 0.5 * x**2


def sum_energy(Path):
    #Function to calculate the sum of energy along a given path.
    #Parameters:    Path (list)
    #Returns:       float: The total energy along the path.
    e = 0
    E = 0
    for i in range(len(Path)-1):
        e = ((m/2) * ((Path[i+1] - Path[i])/dt)**2 
            + potential((Path[i+1] + Path[i])/2))
        E += e
    return E
    