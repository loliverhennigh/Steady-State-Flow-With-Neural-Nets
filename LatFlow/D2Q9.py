
import tensorflow as tf
import numpy as np

# Lattice weights
WEIGHTS = tf.constant([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.], dtype=1)

# Lattice lveloc
LVELOC = tf.constant( [ [0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0]     ], dtype=1)

# Lattice bounce back kernel
BOUNCE = np.array([[ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                   [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                   [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                   [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                   [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0]])
BOUNCE = tf.constant(BOUNCE, dtype=1)

# how to transfer states  
STREAM = np.zeros((3,3,9,9))
STREAM[1,1,0,0] = 1.0
STREAM[1,0,1,1] = 1.0
STREAM[0,1,2,2] = 1.0
STREAM[1,2,3,3] = 1.0
STREAM[2,1,4,4] = 1.0
STREAM[0,0,5,5] = 1.0
STREAM[0,2,6,6] = 1.0
STREAM[2,2,7,7] = 1.0
STREAM[2,0,8,8] = 1.0
STREAM = tf.constant(STREAM, dtype=1)



