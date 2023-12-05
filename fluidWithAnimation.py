from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LBsimAnimated():

    def distance(self, x1 : int, x2: int , y1: int , y2: int):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def calcCurl(self, ux : List[int] , uy : List[int]):
        dfdx = ux[2:,1:-1] - ux[0:-2,1:-1]
        dfxdy = uy[1:-1,2:] - uy[1:-1, 0:-2]
        curl = dfdx - dfxdy
        return curl

    def simulateFluid(self):

        # Simulation Parameters
        Nx = 400        # x resolution
        Ny = 100        # y resolution
        self.tau = .53        # collision time scale
        self.steps = 40000    # numer of time steps
        
        # Lattice Parameters
        self.Nn = 9
        self.vxs = np.array([0,0,1,1,1,0,-1,-1,-1])
        self.vys = np.array([0,1,1,0,-1,-1,-1,0,1])
        self.weights = np.array([4/9, 1/9 , 1/36 , 1/9, 1/36 , 1/9, 1/36 , 1/9, 1/36])
        
        #Initial Conditions
        self.F = np.ones((Ny,Nx,self.Nn)) + .01 * np.random.randn(Ny,Nx,self.Nn) # Need non-uniform values
        self.F[:,:,3] = 2.3 # Have fluid initially moving to the right
        
        #Define obstacle
        self.obstacle = np.full((Ny,Nx), False)
        
        for y in range(Ny):
            for x in range(Nx):
                if (self.distance(Nx//4, x, Ny//2, y) < 15): #Set position and size of obstacle
                    self.obstacle[y][x] = True

        fig = plt.figure()         
        im = plt.imshow(self.calcCurl(np.sum(self.F * self.vxs,2)/np.sum(self.F, 2),np.sum(self.F * self.vys,2) /np.sum(self.F, 2)),cmap='bwr')
        
        # Main Loop
        def update(frame):

            for step in range(self.steps):
                
                #Absorbant Boundary Conditions
                self.F[:, -1, [6,7,8]] = self.F[:, -2, [6,7,8]]
                self.F[:, 0, [2,3,4]] = self.F[:, 1, [2,3,4]]
                
                #Streaming the fluid
                for i, vx, vy in zip(range(self.Nn), self.vxs, self.vys): 
                    self.F[:,:,i] = np.roll(self.F[:,:,i], vx, axis=1)
                    self.F[:,:,i] = np.roll(self.F[:,:,i], vy, axis=0)
                    
                #Calculate fluid moments
                rho = np.sum(self.F, 2) # Density
                ux = np.sum(self.F * self.vxs,2) / rho # Momentum
                uy = np.sum(self.F * self.vys,2) / rho
                
                # Apply Obstacle to fluid system
                boundaryF = self.F[self.obstacle, :]
                boundaryF = boundaryF[:, [0,5,6,7,8,1,2,3,4]] #Reflect fluid off of obstacle
                self.F[self.obstacle, :] = boundaryF
                ux[self.obstacle] = 0
                uy[self.obstacle] = 0
                
                # Collisions
                Feq = np.zeros_like(self.F)
                for i, vx, vy, w in zip(range(self.Nn), self.vxs, self.vys, self.weights):
                    Feq[:,:,i] = rho * w * (1 + 3 * (vx*ux + vy*uy) + 9 * (vx*ux + vy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2)
                
                self.F += -(1/self.tau) * (self.F-Feq)

                im.set_data(self.calcCurl(ux,uy))
                return im,
    
        ani = FuncAnimation(fig, update, blit=True, interval=1, frames=self.steps//200)
        plt.show()

if __name__ == "__main__":
    
    lbsim = LBsimAnimated()
    lbsim.simulateFluid()
