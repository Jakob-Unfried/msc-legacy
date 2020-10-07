import jax.numpy as np

# spin operators and states
s0 = np.array([[1., 0], [0, 1]])
sx = np.array([[0., 1], [1, 0]])
sy = np.array([[0., -1j], [1j, 0]])
sz = np.array([[1., 0], [0, -1]])
sp = np.array([[0., 1.], [0., 0.]])
sm = np.array([[0., 0.], [1., 0.]])
state_x_plus = np.array([1., 1.]) / np.sqrt(2)
state_x_minus = np.array([1., -1.]) / np.sqrt(2)
state_y_plus = np.array([1., 1.j]) / np.sqrt(2)
state_y_minus = np.array([1.j, 1.]) / np.sqrt(2)
state_z_plus = np.array([1., 0])
state_z_minus = np.array([0, 1.])
