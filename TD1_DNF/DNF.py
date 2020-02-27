import numpy as np
import scipy.special as special
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def euclidean_dist(x,y):
    acc=0
    if(len(x) != len(y)):
        return
    for i in range(0, len(x)):
        acc = acc + (x[i] - y[i])**2
    return np.sqrt(acc)
    
def gaussian(dist, sigma):
    return np.exp(-(1/2)*((dist/sigma)**2))

def gaussian_distribution(position, size, sigma):
    mat = np.zeros((size,size))
    for i in range(0, size):
        for j in range(0, size):
            pos = [i/size,j/size]
            dist = euclidean_dist(position, pos)
            mat[i,j] = gaussian(dist, sigma)
    return mat

#a, b position between 0 and 1
def gaussian_activity(a, b, sigma):
    mat = np.zeros((size,size))
    maximum = 0
    for i in range(0, size):
        for j in range(0, size):
            pos = [i/size,j/size]
            dist = euclidean_dist(pos, a)            
            mat[i,j] = gaussian(dist, sigma)
            dist = euclidean_dist(pos, b)
            mat[i,j] = mat[i,j] + gaussian(dist, sigma)
            if mat[i,j] > maximum:
                maximum = mat[i,j]
    return mat
    
def gaussian_distribution_constant(position, size, sigma):
    mat = gaussian_distribution(position, size, sigma)
    return (mat/mat.sum())

# size is size of kernel (double of network)
def generate_selection_kernel(size, coef_excitation, sigma_excitation, global_inhibition):
    kernel = gaussian_distribution_constant([0.5,0.5], size, sigma_excitation)
    kernel = (coef_excitation * kernel) - (global_inhibition / size**2)
    return kernel
    
def update_neuron(position):
    global neurons
    neuron = neurons[position]
    kernelSum = 0
    for i in range (0, size):
        for j in range (0, size):
            sigmoid = special.expit(neurons[i,j])
            i_kernel = size + (i - position[0])
            j_kernel = size + (j - position[1])
            kernelSum += sigmoid * kernel[i_kernel, j_kernel]
    neuronUpdated = neuron + (dt / tau) * ((-neuron) + h + kernelSum + (myInput[position] * gain))
    neurons[position] = neuronUpdated
    

def synchronous_run():
    global neurons
    neurons += (dt/tau) * (-neurons + h + signal.fftconvolve(special.expit(neurons), kernel, mode='same') + (myInput * gain))
    print("iteration: " + str(iteration))
    
# ******************************************************
#                   RUNGE-KUTTA METHODS
# ******************************************************

def func(t,y):
    return 1 - t*y
    
def RK1(y,t):
    return func(t,y)

def RK2(y,t,k1):
    return func(t + dt/2, y + dt/2*k1)
    
def RK3(y,t,k2):
    return func(t + dt/2, y + dt/2*k2)
    
def RK4(y,t,k3):
    return func(t + dt, y + dt*k3)
    
def rungeKutta(y,t):
    rk1 = RK1(y,t)
    rk2 = RK2(y,t,rk1)
    rk3 = RK3(y,t,rk2)
    rk4 = RK4(y,t,rk3)
    return y + dt/6 * (rk1 + 2*rk2 + 2*rk3 + rk4)
    
def euler(y,t):
    return y + dt/6 * RK1(y,t)
    
def synchronous_run_runge_kutta():
    global neurons
    neurons += (dt/tau) * (-neurons + h + rungeKutta(neurons, iteration*dt) + (myInput * gain))
    print("iteration: " + str(iteration))
    
def synchronous_run_euler():
    global neurons
    neurons += (dt/tau) * (-neurons + h + euler(iteration*dt, neurons) + (myInput * gain))
    print("iteration: " + str(iteration))
    

# *****************************************************
#              variables and main functions
# *****************************************************

def updatefig(*args):
    global neurons
    global iteration
    synchronous_run()
    im.set_array(neurons)
    iteration += 1
    return im,

size = 45
tau = 0.5
coef_exc = 15
sigma_exc = 0.05
gi = 250
gain = 5
#h = -4  # suggested value
h = 1  # works well with convolution
dt = 0.1
myInput = gaussian_activity((0.25, 0.25), (0.75, 0.75), sigma_exc)
#myInput = np.random.random((size,size))
neurons = np.zeros((size, size))
kernel = generate_selection_kernel(size*2, coef_exc, sigma_exc, gi)
iteration = 0

if __name__ == '__main__':
    #neurons = signal.fftconvolve(myInput, kernel, mode='same')
    fig = plt.figure()
    im = plt.imshow(myInput, cmap='hot', interpolation='nearest', animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True) # interval est le temps en ms entre chaque frame 
    plt.show()
    
