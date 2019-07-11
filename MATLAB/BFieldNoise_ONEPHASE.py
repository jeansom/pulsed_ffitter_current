import sys, os
import numpy as np
from scipy.integrate import ode


def fun(t, S, Bv0, t_range, g, T2):
    t_ind = np.argmin(np.abs(t-t_range))
    return g*np.cross(S, B[:, t_ind]) - T2*S*np.array([1,1,0])

mag = float(sys.argv[1])
g = float(sys.argv[2])
B0 = float(sys.argv[3])*1e-3
dur_dark = float(sys.argv[4])
pz = float(sys.argv[5])

save_values = []

for trial in range(50):
    sigma = 10**(-mag)
    dt = 1e-2
    T2 = g*sigma**2/B0
    t_range = np.linspace(0, dur_dark, dur_dark/dt)
    Bxn = np.random.normal(sigma, size=len(t_range))
    Byn = np.random.normal(sigma, size=len(t_range))
    Bz = np.ones(len(Bxn))*B0
    B = np.array([ Bxn, Byn, Bz ])
    
    # Create an `ode` instance to solve the system of differential
    # equations defined by `fun`, and set the solver method to 'dop853'.
    solver = ode(lambda t, S: fun(t, S, B, t_range, g, T2))
    solver.set_integrator('dop853')
    
    # Set the initial value z(0) = z0.
    t0 = 0.0
    S0 = [np.sqrt(1-pz**2), 0, pz]
    solver.set_initial_value(S0, t0)
    
    # Create the array `t` of time values at which to compute
    # the solution, and create an array to hold the solution.
    # Put the initial value in the solution array.
    sol = np.empty((len(t_range), 3))
    sol[0] = S0
    
    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t_range[-1]:
        solver.integrate(t_range[k])
        sol[k] = solver.y
        k += 1
    save_values.append(sol[-1])
np.savetxt("Simulations/noiseallsim_mag-"+str(mag)+"_g-"+str(g)+"_B0-"+str(B0)+"_dark-"+str(dur_dark)+"_pz-"+str(pz)+".txt", np.array(save_values))
