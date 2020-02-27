import sys
sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/lunar/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import math
import scipy.special as special
import scipy.signal as signal

'''
Pour installer opencv:
sudo apt-get install opencv* python3-opencv


Si vous avez des problèmes de performances, vous pouvez calculer les convolutions plus rapidement avec :

lateral = signal.fftconvolve(activation, kernel, mode='same')

/!\ Votre kernel doit être de taille impaire pour que la convolution fonctionne correctement (taille_dnf * 2 - 1 par exemple).
'''

images_path = "./pacman/"
image_size = (380, 455)
neurons_size = (455, 380)
window_pos = (50, 280)
window_size = (150, 150)
speed = 1

#~ images_path = "./pacman_60fps/"
#~ image_size = (720,1280)
#~ neurons_size = (1280, 720)

# Pour Opencv : Hue [0,179], Saturation [0,255], Value [0,255]
def selectByColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # créez une carte de saillance en sélectionnant uniquement le jaune
    imgThreshed = np.zeros((image_size[0], image_size[1], 3), dtype = "uint8")
    lowerBound = np.array([30, 200, 200], np.uint8)
    upperBound = np.array([50, 255, 255], np.uint8)
    imgThreshed = cv2.inRange(hsv, lowerBound, upperBound)

    # normalisez ce masque (valeurs entre [0,1])
    cv2.normalize(imgThreshed, imgThreshed, 0, 1, cv2.NORM_MINMAX)
    return imgThreshed


def findCenter(potentials):
    # calculez le centre de gravité de la bulle d'activation du dnf
    mass = 0
    sumX = 0
    sumY = 0
    potentials_normalized = np.zeros(neurons_size)
    cv2.normalize(potentials, potentials_normalized, 0, 1, cv2.NORM_MINMAX)
    for i in range (0, image_size[1]):
        for j in range (0, image_size[0]):
            p = potentials_normalized[i][j]
            mass += p
            sumX += i * p
            sumY += j * p
    if mass == 0:
        halfsize = tuple(s/2 for s in window_size)
        center = tuple(map(lambda x, y: x + y, window_pos, halfsize)) # center of the window
    else:
        center = (math.floor(sumY/mass), math.floor(sumX/mass))
    return center


# speed between 0 and 1
def moveWindow(center, speed):
    # déplacez graduellement la fenêtre d'attention pour placer le centre de gravité du dnf au centre de celle-ci
    global window_pos
    global neurons
    halfsize = tuple(s/2 for s in window_size)
    window_center = tuple(map(lambda x, y: x + y, window_pos, halfsize))
    move = tuple(map(lambda x, y: x - y, center, window_center))
    move = tuple(m*speed for m in move) # 0 <= speed <= 1
    window_pos = tuple(map(lambda x, y: math.floor(x + y), window_pos, move))
    w = list(window_pos)
    for i in (0,1):
        if w[i] < 0:
            w[i] = 0
        if w[i] > image_size[i] - window_size[i]:
            w[i] = image_size[i] - window_size[i]
    window_pos = tuple(w)

def track(frame):
    global myInput
    global neurons
    # dnf.input = input
    myInput = selectByColor(frame)
    # dnf.update_map()
    synchronous_run()
    # cv2.imshow("Input", dnf.input)
    cv2.imshow("Input", myInput*255)
    # cv2.imshow("Potentials", dnf.potentials)
    potentials = neurons.copy()
    cv2.normalize(neurons, potentials, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Potentials", potentials)
    # center = findCenter(dnf.potentials)
    center = findCenter(neurons)
    # moveWindow(center, speed)
    moveWindow(center, speed)
    
# *********************************************
#                 DNF
# *********************************************

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
    mat = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            pos = [i/size[0],j/size[1]]
            dist = euclidean_dist(position, pos)
            mat[i,j] = gaussian(dist, sigma)
    return mat

def gaussian_distribution_constant(position, size, sigma):
    mat = gaussian_distribution(position, size, sigma)
    return (mat/mat.sum())

def generate_selection_kernel(size, coef_excitation, sigma_excitation, global_inhibition):
    kernel = gaussian_distribution_constant([0.5,0.5], size, sigma_excitation)
    kernel = (coef_excitation * kernel) - (global_inhibition / size[0]*size[1]*2)
    return kernel

def synchronous_run():
    global neurons
    neurons += (dt/tau) * (-neurons + h + signal.fftconvolve(special.expit(neurons), kernel, mode='same') + (myInput * gain))

tau = 0.5
coef_exc = 15
sigma_exc = 0.05
gi = 250
gain = 5
h = -4
dt = 0.1
neurons = np.zeros(neurons_size)
kernel = generate_selection_kernel(tuple(i*2-1 for i in neurons_size), coef_exc, sigma_exc, gi)
myInput = np.zeros(neurons_size)

# *********************************************
#                 DNF - END
# *********************************************



if __name__ == '__main__':
    frame = cv2.imread(images_path+"pacman00001.png")
    # TODO : initialisez votre DNF ici
    # TEST
    #~ img = selectByColor(frame)
    #~ cv2.imshow('pacman',img)
    #~ cv2.waitKey(0) # waits until a key is pressed

    for i in range(1, 196):  #pacman
    #~ for i in range(135, 1000): #pacman60fps
        frame = cv2.imread(images_path + "pacman{0:05d}.png".format(i))
        track(frame)
        frame_np = np.asarray(frame)
        window = frame_np[window_pos[1]:window_pos[1]+window_size[1], window_pos[0]:window_pos[0]+window_size[0]]
        cv2.imshow("Window", window)
        key = cv2.waitKey(500)
        if key == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
