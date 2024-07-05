import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import os
import csv

xml_path = 'Kamal_final.xml' #xml file (assumes this is in the same folder as this file)
simend = 7 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


t = []
theta = []
theta_dot = []

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    t.append(data.time)
    theta.append(data.sensordata[0])
    theta_dot.append(data.sensordata[1])

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures

model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Get the index of the sensors
# theta_sensor_index = model.sensor_name2id('j1_pos')
# theta_dot_sensor_index = model.sensor_name2id('j1_vel')

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        # File path to save CSV
        # csv_file = 'data.csv'

        # Combine arrays into a list
        # theta = np.degrees(theta) + 45
        theta = theta + np.radians(45)
        # theta_dot = np.degrees(theta_dot)
        data = [t, theta, theta_dot]
        print(len(theta))
        # data_rearranged = np.column_stack(data)

        # Save rearranged data to CSV
        # np.savetxt(csv_file, data_rearranged, delimiter=',')
        # print(theta)
        # print(theta_dot)

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(t,theta)
        # plt.ylabel('theta')
        # plt.subplot(2,1,2)
        # plt.plot(t,theta_dot)
        # plt.ylabel('theta_dot')
        # plt.show()
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()






# Parameters
g = 9.81 # acceleration due to gravity (m/s^2)
L = np.sqrt(.9**2 + .9**2) # length of the pendulum (m)
def pendulum_dynamics(t, y):
    Theta, omega = y
    dTheta_dt = omega
    domega_dt = - (g / L) * np.sin(Theta)
    return np.array([dTheta_dt, domega_dt])

# RK4 implementation
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Initial conditions
Theta0 = theta[0]  # initial angle (radians)
omega0 = theta_dot[0]        # initial angular velocity (radians/s)
y0 = np.array([Theta0, omega0])

# Time span for the simulation
t_span = (0, simend)  # 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], len(theta))  # 500 points for evaluation
dt = t_eval[1] - t_eval[0]  # time step

# Solve the system using RK4
y = np.zeros((len(t_eval), 2))
y[0] = y0

for i in range(1, len(t_eval)):
    y[i] = rk4_step(pendulum_dynamics, t_eval[i-1], y[i-1], dt)

# Extract the solution
Theta = y[:, 0]
omega = y[:, 1]

# Convert Theta from radians to degrees and add 45 degrees
Theta_deg = np.degrees(Theta) + 45

# Convert omega from radians per second to degrees per second
omega_deg_s = np.degrees(omega)

# Plot the results
plt.figure(figsize=(12, 5))

# Plot Theta vs time
plt.subplot(1, 2, 1)
# plt.plot(t_eval, Theta, label='Theta (degrees)')
# plt.plot(t_eval, theta, label='theta (degrees)')
plt.plot(t_eval, theta - Theta, label='theta (degrees)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (degrees)')
plt.title('Angular Displacement vs Time')
plt.legend()

# Plot omega vs time
# plt.subplot(1, 2, 2)
# plt.plot(t_eval, omega_deg_s, label='Omega (deg/s)', color='r')
# plt.xlabel('Time (s)')
# plt.ylabel('Omega (deg/s)')
# plt.title('Angular Velocity vs Time')
# plt.legend()

plt.tight_layout()
plt.show()
print(np.sqrt(np.mean((Theta - theta) ** 2)))
# print(np.subtract(Theta,theta))

