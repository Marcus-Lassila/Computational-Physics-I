import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import time
import sys

def find_optimal_throws(initial_speed, initial_height, m, g=9.81, b_range=(0.0, 0.15, 75), time_range=(0,3,3000), theta_range=(0,90,900), quadratic=False, running_time = False):
    '''
    This function simulates the throw of a tennis ball in a media with a drag force linear or quadratic in velocity, and finds the angle for the throw which results
    in the longest throw as well as the distance of the longest throw. 

    Arguments:
     
    initial_speed (float): initial speed of the tennis ball in meters per second.

    initial_height (float): initial height from which the tennis ball is thrown in meters.

    m (float): mass of the tennis ball in kg.

    Optional arguments:

    g (float): gravitational acceleration in meters per second squared. Default value of 9.81 meters per second squared.
    
    b_range (tuple): (b_min, b_max, num_samples), specify range for the drag constant b. The samples are spaced evenly.

    time_range (tuple): (start time, stop time, num_samples), specify start time, stop time, and number of evenly spaced time samples. 

    theta_range (tuple): (theta_min, theta_max, num_samples), specify the range of the trowing angle (in degrees) and number of evenly spaced samples.

    quadratic (bool): If quadratic=True the drag force is quadratic in velocity, else it is assumed to be linear in velocity.

    running_time (bool): if true the function returns the running time of the function in seconds.

    Return an array with the distance of the longest throw (in meters) and the corresponding throwing angle (in degrees) for each value of the drag force constant.
    '''
    # Start timing. 
    t0 = time.time()
    print("Computing trajectories. This might take a while...")

    def diffeq(z,t,m,g,b):
        '''
        Return the derivative of z = [x, y, dx/dt, dy/dt] using the equation of motion
        for the tennis ball to evaluate the derivatives of dx/dt and dy/dt. 
        The equation of motion is Newtons 2nd equation in the presence of a gravitational
        force in the negative y-direction and a drag force linear in velocity.

        t = array of time samples. 

        m = mass of the ball in kg.

        g = gravitational acceleration in m/s^2.

        b = constant of proportionality for the drag force in kg m/s.
        '''
        if quadratic:
            # The sign is included so that the drag force always is opposite of the direction of motion.
            return [z[2], z[3], -b/m*z[2]*np.sqrt(z[2]**2 + z[3]**2), -g-b/m*z[3]*np.sqrt(z[2]**2 + z[3]**2)]
        return [z[2], z[3], -b/m*z[2], -g-b/m*z[3]]

    
    t_start, t_stop, time_samples = time_range
    b_min, b_max, b_samples = b_range
    theta_min, theta_max, theta_samples = theta_range
    
    # Convert to radians
    rad_to_deg = 180/np.pi
    theta_min /= rad_to_deg
    theta_max /= rad_to_deg

    # Define time interval.
    t = np.linspace(t_start, t_stop, time_samples)

    # Initial conditions for position [x0, y0]
    initial_position = [0.0, initial_height]

    # Always throw with the same initial speed.
    v = initial_speed
    theta = np.linspace(theta_min, theta_max, theta_samples) 
    vx = v*np.cos(theta)
    vy = v*np.sin(theta)

    # Initialize numpy arrays for storing distances for different throws.
    distance = np.zeros(theta_samples)

    # Define array to store different values for the drag constant b.
    b = np.linspace(b_min,b_max,b_samples)

    # Define array to store results for throwing distance.
    res = np.zeros((b_samples,3))

    # Iterate over different values of b, the constant of proportionality for the linear drag force term in the eom.
    for i in range(b_samples):
        # Iterate over different throwing angles.
        for j in range(theta_samples):
            # Define initial conditions
            z0 = initial_position + [vx[j],vy[j]]
            # Solve ode using scipy function odeint.
            z = odeint(diffeq,z0,t,args=(m,g,b[i]))
            x = z[:,0]
            y = z[:,1]
            xp = x[y>=0]
            yp = y[y>=0]
            # (x1,y1) is the point above the x-axis which is closest to the x-axis.
            # (x2,y2) is the point below the x-axis which is closest to the x-axis.
            x1 = x[len(xp)-1]
            y1 = y[len(yp)-1]
            try:
                x2 = x[len(xp)]
                y2 = y[len(yp)]
            except IndexError:
                print("The time interval is too short to capture the entire trajectory. Aborting...")
                sys.exit()
            # Calculate the x-coordinate where the straight line between (x1,y1) and (x2,y2) cross the x-axis.
            distance[j] = x1 - (x2 - x1)/(y2 - y1)*y1   
        # Find the optimal angle and corresponding throwing distance given the current value of b.
        optimal_angle = theta[distance.argmax()]*rad_to_deg
        longest_distance = distance.max()
        res[i,:] = b[i], optimal_angle, longest_distance
    # End timing
    rt = time.time() - t0
    if running_time:
        return res, rt
    return res 

def plot_optimal_throws(data,description):
    '''
    Plot the data in data using pyplot and add the corresponding description to the legend.

    data (list): a list of numpy arrays as that returned from the find_optimal_trows() function.

    description (list): a list of strings used to describe the data in the legend of the plot.
    '''
    
    # Plot optimal throwing angle as a function of the drag force constant. 
    for i in range(len(data)):
        plt.plot(data[i][:,0],data[i][:,1])
    plt.legend(description)
    plt.xlabel("Drag force constant " r'$b$' " [kg m/s]")
    plt.ylabel("Optimal throwing angle " r'$\theta_c$' " [deg]")
    plt.yticks(np.arange(0,50,5))
    plt.show()

    # Plot the maximal throwing distance as a function of the drag force constant.
    for i in range(len(data)):
        plt.plot(data[i][:,0],data[i][:,2])
    plt.legend(description)
    plt.xlabel("Drag force constant " r'$b$' " [kg m/s]")
    plt.ylabel("Maximal throwing distance " r'$d_{max}}$' " [m]")
    plt.yticks(np.arange(0,20,2))
    plt.show()

def main():

    # m is the mass of the tennis ball in kg. The value is measured from an actuall tennis ball. 
    m = 0.058
    
    #Reasonable initial speed of 10 meters per second.
    initial_speed = 10.0
    
    data = []
    description = []

    # Store some parameter values for different initial heights in a list of tuples (height, t_max, theta_min, theta_max).
    # Values for t_max, theta_min and theta_max is found from a fast running of the program at lower number of samples.
    # Then it is possible to run the program a bit faster at higher number of samples as the relevant intervals for time and angle can be selected directly.
    # Parameters for the quadratic drag force:
    parameters = [(0.0, 2, 32, 45), (2.0, 2, 20, 42), (6.0, 4, 13, 35), (12.0, 7, 8, 30)] 

    # Parameters for the linear drag force:
    # parameters = [(0.0, 3, 25, 45), (2.0, 3, 8, 42), (6.0, 3, 0, 36), (12.0, 5, -2, 30)]
    # It is safe to increase number of theta samples in the linear case since the calculations are faster here.

    total_time = 0.0
    for height, t_max, theta_min, theta_max in parameters:
        res, rt = find_optimal_throws(initial_speed, height, m, time_range=(0, t_max, t_max*10), theta_range=(theta_min, theta_max, 20), quadratic=True, running_time=True)
        data.append(res)
        description.append("initial height: " + str(height) + " meters")
        if rt < 60.0:
            print("Running time: " + str(rt) + " seconds")
            print()
        else:
            print("Running time: " + str(rt/60) + " minutes")
            print()
        total_time += rt
    if total_time < 60.0:
        print("Total running time: " + str(total_time) + " seconds")
    else:
        print("Total running time: " + str(total_time/60) + " minutes")

    plot_optimal_throws(data,description)
    
if __name__=='__main__':
    main()