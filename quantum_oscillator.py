import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm

class QuantumOscillator(object):
    '''A QuantumOscillator object models a quantum harmonic or anharmonic oscillator
    in an infinite rectangular well potential.'''

    def __init__(self, Lx, Ly, Nx, Ny, kx, ky, g=1, hbar=1, mass=1):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.kx = kx
        self.ky = ky
        self.g = g
        self.hbar = hbar
        self.mass = mass

    # Some getters and setters:

    def get_size(self):
        '''Return size of the rectangle as a tuple (Lx, Ly).'''
        return (self.Lx, self.Ly)
    
    def get_grid_size(self):
        '''Return the size of the grid as a tuple (Nx, Ny)'''
        return (self.Nx, self.Ny)

    def get_hbar(self):
        '''Return the value used for Planck's constant hbar.'''
        return self.hbar
    
    def set_hbar(self, hb):
        '''Set the value for Planck's constant hbar.'''
        self.hbar = hb

    def get_mass(self):
        '''Return the mass of the quantum oscillator.'''
        return self.mass

    def set_mass(self, m):
        '''Set the mass of the quantum oscillator.'''
        self.mass = m

    def get_k(self):
        '''Return 'spring constants' for the harmonic oscillators in the x and y direction as a tuple (kx, ky).'''
        return (self.kx, self.ky)

    def set_k(self, kx, ky):
        '''Set the 'spring constants' for the harmonic oscillator.'''
        self.kx = kx
        self.ky = ky
    
    def get_g(self):
        '''Return the coupling constant g for the anharmonic term.'''
        return self.g

    def set_g(self, g):
        '''Set the anharmonic coupling constants.'''
        self.g = g

    def laplacian(self):
        '''Return a sparse matrix representing the Laplacian operator in the finite difference method.'''
        Lx, Ly = self.Lx, self.Ly
        Nx, Ny = self.Nx, self.Ny
        dx, dy = Lx/Nx, Ly/Ny

        x1 = np.ones(Nx*Ny)/(dx**2)
        x2 = x1.copy()
        # Set some of the elements to zero in order to decouple consecutive rows.
        for i in range(1, Ny):
            x1[i*Nx-1] = 0
            x2[i*Nx] = 0
        y = np.ones(Nx*Ny)/(dy**2)
        d = np.ones(Nx*Ny)*(-2/(dx**2) - 2/(dy**2))
        offsets = np.array([-Ny, -1, 0, 1, Ny])
        data = np.array([y, x1, d, x2, y])
        # Return a sparse diagonal matrix with data on offset diagonals.
        return dia_matrix((data, offsets), shape=(Nx*Ny, Nx*Ny)) 

    def potential(self, plot=False, anharmonic=False):
        '''
        Return a sparse matrix representing the potential of the quantum system.
        
        Optional arguments:
        
        plot (bool): If True the potential is plotted as a 2D surface.

        anharmonic (bool): If False the potential is a harmonic potential (quadratic in position) centered at the middle of the rectangle.
        If True anharmonic terms 'quartic' (4:th power) in position are added. 
        '''
        Lx, Ly = self.Lx, self.Ly
        Nx, Ny = self.Nx, self.Ny
        dx, dy = Lx/Nx, Ly/Ny

        if anharmonic: # Create the anharmonic potential...
            V = np.array([[(self.kx/2*(i*dx-(Lx-dx)/2)**2 + self.ky/2*(j*dy-(Ly-dy)/2)**2 + self.g/4*((i*dx-(Lx-dx)/2)**2 + (j*dy-(Ly-dy)/2)**2)**2) for j in range(Ny)] for i in range(Nx)])
        else: # Create a harmonic potential...
            V = np.array([[(self.kx/2*(i*dx-(Lx-dx)/2)**2 + self.ky/2*(j*dy-(Ly-dy)/2)**2) for j in range(Ny)] for i in range(Nx)])
        if plot: # Plot the potential as a surface plot.
            X = np.linspace(0, Lx, Nx)
            Y = np.linspace(0, Ly, Ny)
            Y, X = np.meshgrid(Y, X)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, V, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
            plt.show()
        return dia_matrix((V.flatten(),[0]), shape=(Nx*Ny, Nx*Ny))

    def hamiltonian(self, anharmonic=False):
        '''Return the Hamiltonian of the QuantumOscillator object as a sparse matrix.'''
        return -self.hbar**2/(2*self.mass)*self.laplacian() + self.potential(anharmonic=anharmonic)

    def solve_shrodingers_equation(self, number_of_states=10, anharmonic=False):
        '''
        Solve Shrödinger's equation to find the lowest energy states of the quantum system.
        Uses the scipy function eigsh to solve the eigenvalue problem for a sparse and hermitian matrix. 
        
        Return eigenvalues, eigenstates.

        Optional arguments:

        number_of_states (int): The number of lowest energy states to be returned.

        anharmonic (bool): If True a anharmonic term (quartic in position) is included in the potential.
        Otherwise the Hamiltonian is that of a harmonic oscillator.
        '''
        evals, evecs = eigsh(self.hamiltonian(anharmonic=anharmonic), k=number_of_states, which='SM')
        return evals, evecs 

    def plot_eigenvectors(self, number_of_states=10, anharmonic=False):
        '''
        Plot a few of the lowest eigenstates of the quantum system.
        Also prints a list of the corresponding energy levels.
        '''
        evals, evecs = self.solve_shrodingers_equation(number_of_states=number_of_states, anharmonic=anharmonic)
        print("The lowest energy levels: ")
        print(list(evals))
        X = np.linspace(0, self.Lx, self.Nx)
        Y = np.linspace(0, self.Ly, self.Ny)
        Y, X = np.meshgrid(Y, X)
        for i in range(len(evals)):
            Z = evecs[:,i].reshape(self.Nx,self.Ny)
            # Make a surface plot...
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            plt.show()
            # Make a filled contour plot...
            plt.contourf(X, Y, Z, levels=30, cmap=cm.coolwarm)
            plt.colorbar()
            plt.show()

def main():
    '''Compute the lowest energy levels and corresponding states for a quantum (an)harmonic oscillator
    in a square infinite potential well.'''
    
    Lx = Ly = 8
    Nx = Ny = 300 # Takes some seconds to solve the eigenvalue problem with this value.
    qs = QuantumOscillator(Lx,Ly,Nx,Ny,1,1,g=1)
    # Set anharmonic=False for a harmonic oscillator
    qs.potential(plot=True, anharmonic=True)
    qs.plot_eigenvectors(anharmonic=True)

if __name__=='__main__':
    main()