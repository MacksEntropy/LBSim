import numpy as np
import matplotlib.pyplot as plt

def distance(x1,x2,y1,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def calcCurl(ux,uy):
    dfdx = ux[2:,1:-1] - ux[0:-2,1:-1]
    dfxdy = uy[1:-1,2:] - uy[1:-1, 0:-2]
    curl = dfdx - dfxdy
    return curl

def simulateFluid():
    # System Constants
    Nx = 400
    Ny = 100
    tau = .6
    steps = 4000
    
    # Lattice parameters
    Nn = 9
    vxs = np.array([0,0,1,1,1,0,-1,-1,-1])
    vys = np.array([0,1,1,0,-1,-1,-1,0,1])
    weights = np.array([4/9, 1/9 , 1/36 , 1/9, 1/36 , 1/9, 1/36 , 1/9, 1/36])
    
    #Initial Conditions of fluid F
    F = np.ones((Ny,Nx,Nn)) + .01 * np.random.randn(Ny,Nx,Nn) # Need non-uniform values
    F[:,:,3] = 2.3 # Have fluid initially moving to the right
    
    #Define obstacle, initially a cylinder
    obstacle = np.full((Ny,Nx), False)
    
    for y in range(Ny):
        for x in range(Nx):
            if (distance(Nx//4, x, Ny//2, y) < 13): #Set position and size of obstacle
                obstacle[y][x] = True
                
    # Main Loop
    for step in range(steps):
        
        #Absorbant Boundary Conditions
        F[:, -1, [6,7,8]] = F[:, -2, [6,7,8]]
        F[:, 0, [2,3,4]] = F[:, 1, [2,3,4]]
        
        #Streaming the fluid
        for i, vx, vy in zip(range(Nn), vxs, vys): 
            F[:,:,i] = np.roll(F[:,:,i], vx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], vy, axis=0)
            
        #Calculate fluid moments
        rho = np.sum(F, 2) # Density
        ux = np.sum(F * vxs,2) / rho # Momentum
        uy = np.sum(F * vys,2) / rho
        
        # Apply Obstacle to fluid system
        boundaryF = F[obstacle, :]
        boundaryF = boundaryF[:, [0,5,6,7,8,1,2,3,4]] #Reflect fluid off of obstacle
        F[obstacle, :] = boundaryF
        ux[obstacle] = 0
        uy[obstacle] = 0
        
        # Collisions
        Feq = np.zeros_like(F)
        for i, vx, vy, w in zip(range(Nn), vxs, vys, weights):
            Feq[:,:,i] = rho * w * (1 + 3 * (vx*ux + vy*uy) + 9 * (vx*ux + vy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2)
        
        F += -(1/tau) * (F-Feq)

        plot_every = 10
        if (step % plot_every == 0):

            # Plot magnitude of momentum 
            # plt.imshow(np.sqrt(ux**2 + uy**2)) 

            # Plot vorticity
            curl = calcCurl(ux,uy)
            plt.imshow(curl, cmap="bwr")

            plt.pause(0.1)
            plt.cla()

if __name__ == "__main__":
    simulateFluid()
