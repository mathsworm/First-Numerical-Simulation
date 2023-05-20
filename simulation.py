import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import math

start = time.time()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

n_particles = 30  # number of particles for the simulation
n_springs = (n_particles*(n_particles - 1))//2  # all particles connected by springs to each other

particles = np.zeros(n_particles, dtype=[('mass', float),
                                      ('position', float, (2,)),
                                      ('velocity', float, (2,)),
                                      ('force', float, (2,)),
                                      ('size', float),
                                      ('growth', float),
                                      ('color', float, (4,))])

particles['mass'] = np.ones(n_particles)  # all masses are equal to 1
 
for i in range(n_particles):  # simulate a macroscopic object as a set of microscopic objects
    particles['position'][i][0] = 0.5 + 0.2 * math.cos(2*i*math.pi/n_particles)
    particles['position'][i][1] = 0.8 + 0.2 * math.sin(2*i*math.pi/n_particles)   

particles['size'] = 50*np.ones(len(particles))  

springs = np.zeros(n_springs, dtype = [('p1', int),
                                       ('p2', int),
                                       ('nl', float),
                                       ('k_s', float),
                                       ('k_d', float)])

curr = 0
for i in range (n_particles):
    for j in range (i+1,n_particles):
        springs['p1'][curr] = i
        springs['p2'][curr] = j
        springs['nl'][curr] = np.linalg.norm(particles['position'][i] - particles['position'][j])  # natural length of spring is the distance
        springs['k_s'][curr] = 0.3   # found a good value of k_s by trial and error
        springs['k_d'][curr] = 0.07  # found a good value of k_d by trial and error
        curr += 1

gravity = np.array([0,-0.005])
ground = 0.1

scat = ax.scatter(particles['position'][:, 0], particles['position'][:, 1],
                  s=particles['size'], lw=0.5)

def update(frame_number):

    if time.time() - start > 10:
        exit()

    dt = 0.1

    for i in range(len(particles)):
        particles['force'][i] = particles['mass'][i] * gravity

    for i in range (len(springs)):
        # unpack all data of the spring
        p1,p2,nl,k_s,k_d = springs['p1'][i], springs['p2'][i], springs['nl'][i], springs['k_s'][i], springs['k_d'][i]  
 
        p1_pos = particles['position'][p1]
        p2_pos = particles['position'][p2]
        p1_vel = particles['velocity'][p1]
        p2_vel = particles['velocity'][p2]

        vec = p1_pos - p2_pos  # relative position 
        vel = p1_vel - p2_vel  # relative velocity
        mag = np.linalg.norm(vec)  # distance

        force = (k_s*((mag/nl)-1.0) + k_d * np.dot(vel/nl,vec/mag))*(vec/mag)  # force between pairs of particles

        particles['force'][p1] -= force
        particles['force'][p2] += force

    for i in range(len(particles)):
        particles['velocity'][i] += dt*(particles['force'][i]/particles['mass'][i])
        particles['position'][i] += dt*(particles['velocity'][i])

        if particles['position'][i][1] <= ground:
            particles['position'][i][1] = 2*ground - particles['position'][i][1]
            particles['velocity'][i][1] *= -1

    scat.set_sizes(particles['size'])
    scat.set_offsets(particles['position'])

anim = FuncAnimation(fig, update, interval=10)
plt.show()
