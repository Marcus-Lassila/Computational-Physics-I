import numpy as np
from matplotlib import pyplot as plt
import random
import cProfile
import time
import sys

class IsingModel(object):
    '''An IsingModel object consists of a 2D lattice S of spins, interacting only with
    their nearest neighbours and with a constant interaction strength J. The temperature
    of the system is kT and B is the strength of the external magnetic field.    
    '''

    def __init__(self, size, B=0, kT=1, J=1):
        self.size = size
        self.B = B
        self.kT = kT
        self.J = J
        self.S = self.random_lattice()
          
    def random_lattice(self):
        '''Create a 2D numpy array containing random integers -1 or 1.'''
        S = np.floor(np.random.random(self.size)*2)*2 - 1
        return S.astype(int)

    def spin_up_lattice(self):
        '''Create a spin lattice containing only up spins (+1).'''
        S = np.zeros((self.size)) + 1
        return S.astype(int)

    # Some getters and setters:

    def get_size(self):
        '''Return lattice size.'''
        return self.size
    
    def get_B(self):
        '''Return external magnetic field.'''
        return self.B
    
    def set_B(self, B):
        '''Set external magnetic field.'''
        self.B = B

    def get_J(self):
        '''Return interaction strength.'''
        return self.J
    
    def set_J(self, J):
        '''Set interaction strength.'''
        self.J = J

    def get_kT(self):
        '''Return temperature.'''
        return self.kT
    
    def set_kT(self, kT):
        '''Set temperature.'''
        self.kT = kT

    def get_S(self):
        '''Retun spin lattice.'''
        return self.S
    
    def set_S(self,S):
        '''Set spin lattice to S'''
        self.S = S

    # Other methods:

    def neighbours(self,i,j):
        '''Return the positions of the neighbours of the spin at position i, j,
        assuming periodic boundary conditions.'''
        M, N = self.size
        assert 0 <= i < M and 0 <= j < N

        left   = ((i-1)%M, j)
        right  = ((i+1)%M, j)
        top    = (i, (j-1)%N)
        bottom = (i, (j+1)%N)
        return left, right, top, bottom

    def energy(self):
        '''Return the average energy per spin of the 2D lattice.'''
        S = self.S
        M, N = self.size
        E = - self.B*sum(sum(S))
        for i in range(M):
            for j in range(N):
                E -= self.J/2*S[i,j]*sum([S[k,l] for k,l in self.neighbours(i,j)]) 
        return E/(M*N)

    def magnetization(self):
        '''Return the average magnetization per spin of the 2D lattice.'''
        M, N = self.size
        return sum(sum(self.S))/(M*N)

    def plot_lattice(self):
        '''Plot the 2D lattice of spins.'''
        plt.imshow(self.S, cmap='binary')
        plt.show()

    def metropolis(self, flips, calc_magnetization=False):
        '''
        An implementation of the metropolis algorithm. Picks a random spin and flips it.
        The flip is kept with a probability equal to exp(-de/kT), where 'de' is the change in the energy of the lattice by flipping the spin.
        The process is repeated flips number of times.
        
        Arguments:

        flips (int): The number of times to pick a random spin and flip it. Note that flips is unlikely the number of flips that are kept.

        Optional:

        calc_magnetization (bool): If true, the function calculates the average magnetization per spin for every M*N iteration, where the lattice has size=(M,N).
        Return an array of the average magnetization per spin at these different steps in the evolution of the lattice. Return an AssertionError if flips < M*N.     

        '''
        M, N = self.size
        B, kT, J = self.B, self.kT, self.J

        if calc_magnetization:
            assert flips >= M*N
            # Average magnetization per spin is calculated for only once
            # for every M*N number of flips to reduce computational time.
            avr_flips_per_spin = flips//(M*N)
            avg_mag = np.zeros(avr_flips_per_spin)

        for m in range(flips):

            # Pick a random spin and calculate its contribution to the energy.
            i, j = random.randint(0, M-1), random.randint(0, N-1)

            # Calculate the contribution to the energy by the spin at i,j before and after flipping it.
            # Note that we have multiplied the J-term by a factor of two to account for both 
            # the interaction of the spin at i,j with its neighbours and the interaction of the neighbours
            # with the spin at i,j.
            # e = - B*self.S[i,j] - J*self.S[i,j]*sum([self.S[k,l] for k,l in self.neighbours(i,j)])
            # enew = + B*self.S[i,j] + J*self.S[i,j]*sum([self.S[k,l] for k,l in self.neighbours(i,j)])
            # de = enew - e

            de = 2*B*self.S[i,j] + 2*J*self.S[i,j]*sum([self.S[k,l] for k,l in self.neighbours(i,j)])

            # Determine if flip should be kept.
            if random.random() < np.exp(-de/kT):
                self.S[i,j] = - self.S[i,j]
            
            if calc_magnetization and m%(M*N) == 0:
                avg_mag[m//(M*N)] = self.magnetization()

        if calc_magnetization:
            return avg_mag


def main():
    '''Run simulations of the 2D Ising model using the IsingModel Class.'''

    def plot_magnetization_vs_kT(size, flips, simuls, kT_range):
        '''
        Plot absolute value of the average magnetization per spin vs temperature kT.
        
        Arguments:

        size (int, int): Dimensions of spin lattice.

        flips (int): Number of spin flips for the Metropolis algorithm.

        simuls (int): Number of times the evolution of the system is simulated per value of kT.
        The average magnetization per spin is averaged over each simulation per value of kT. 

        kT_range (int, int, int): kT interval to simulate over; kT_range=(min, max, steps).
        '''

        # Start timing the function.
        t0 = time.time()

        ising_model = IsingModel(size)

        kT_min, kT_max, kT_steps = kT_range
        kT = np.linspace(kT_min, kT_max, kT_steps)

        avg_mag = np.zeros(kT_steps)

        for i in range(kT_steps):
            
            if i%(kT_steps//10) == 0:
                print(str((i*100)//kT_steps)+"% done...")

            ising_model.set_kT(kT[i])
            mag_samples = np.zeros(simuls)
        
            for j in range(simuls):
                
                if kT[i] < 2:
                    # Evolve the lattice using the Metropolis algorithm.
                    ising_model.metropolis(flips//16)

                    # Calculate the absolute value of the average magnetization per spin
                    # and store the result in the sample array.
                    mag_samples[j] = abs(ising_model.magnetization())

                    # Set all spins to spin-up.
                    ising_model.set_S(ising_model.spin_up_lattice())
                else:

                    # Evolve the lattice using the Metropolis algorithm.
                    ising_model.metropolis(flips)

                    # Calculate the absolute value of the average magnetization per spin
                    # and store the result in the sample array.
                    mag_samples[j] = abs(ising_model.magnetization())

                    # Randomize the spin lattice for the next simulation.
                    ising_model.set_S(ising_model.random_lattice())

            # Calculate the average of the magnetization samples.
            avg_mag[i] = sum(mag_samples)/simuls
            
        print("100% done!")
        # Stop timing and print time in minutes
        t1 = time.time() - t0
        print("Total time for running the simuls: " + str(t1/60) + " minutes")

        # Plot the average of the average magnetization per spin vs the temperature with '*'. 
        plt.plot(kT,avg_mag,'*')
        plt.xlabel("Temperature " r'$kT$' " [AU]")
        plt.ylabel("Average magnetization per spin " r'$\left\langle M \right\rangle}$' " [AU]")
        plt.show()

        # Plot the average of the average magnetization per spin vs the temperature with continious line. 
        plt.plot(kT,avg_mag)
        plt.xlabel("Temperature " r'$kT$' " [AU]")
        plt.ylabel("Average magnetization per spin " r'$\left\langle M \right\rangle}$' " [AU]")
        plt.show()


    def fraction_in_equilibrium(size, avr_flips, simuls, B=0, kT=1, J=1):
        '''
        Return the fraction of Metropolis simulations reaching the (anti-)ferromagnetic equilibrium state.
        
        Arguments:

        size (int, int): Dimensions of spin lattice.

        flips (int): Number of spin flips for the Metropolis algorithm.

        simuls (int): Number of times the evolution of the system is simulated and the average magnetization per spin is calculated.

        Optional:

        B (float/int): External magnetic field of the Ising model.

        kT (float/int): Temperature of the Ising model.

        J (float/int): Spin interaction strength of the Ising model.

        '''

        # Initialize an Ising model object
        ising_model = IsingModel(size, B=B, kT=kT, J=J)

        avr_mag = np.zeros(simuls)
        for i in range(simuls):

            # Evolve the lattice using the Metropolis algorithm.
            ising_model.metropolis(flips)

            # Calculate the absolute value of the average magnetization per spin
            # and store the result in the avr_mag array.
            avr_mag[i] = abs(ising_model.magnetization())
            
            # Randomize the lattice before starting the next simulation.
            ising_model.set_S(ising_model.random_lattice())
        
        success = [x for x in avr_mag if abs(x)>0.95] 
        return len(success)/simuls


    # Code for finding the average number of flips per spin needed to reach equilibrium at kT=1:

    M = N = 12
    size = (M,N)
    avr_flips_per_spin = 400
    flips = avr_flips_per_spin*M*N
    simulations = 100

    res = fraction_in_equilibrium(size, flips, simulations)
    # With these inputs the fraction reaching equilibrium is about 99%.
    print("Fraction of simulations reaching equilibrium at kT = 1: " + str(res))

    ising_model = IsingModel(size)
    avr_mag = ising_model.metropolis(flips, calc_magnetization=True)

    plt.plot(np.linspace(0,avr_flips_per_spin,avr_flips_per_spin),avr_mag)
    plt.xlabel("Average number of flips per spin")
    plt.ylabel("Average magnetization per spin " r'$\left\langle M \right\rangle}$' " [AU]")
    plt.show()

    # Code for plotting the absolute value of the average magnetization per spin vs temperature:

    M = N = 12
    size = (M,N)
    avr_flips_per_spin = 800
    flips = avr_flips_per_spin*M*N
    simulations = 100
    kT_range = (0.1, 4, 30)

    # This takes a long time to run...
    plot_magnetization_vs_kT(size, flips, simulations, kT_range)


    # Code for plotting a large lattice before and after evolving the lattice with the Metropolis algorithm:

    ising_model = IsingModel((1000,1000))
    ising_model.plot_lattice()

    # A large lattice requires a large number of flips before any evolution can be seen.
    ising_model.metropolis(10**8)
    ising_model.plot_lattice()

 
if __name__=='__main__':
    main()