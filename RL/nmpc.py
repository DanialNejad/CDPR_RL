import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import casadi as ca
import numpy as np
import os
import csv

xml_path = 'assets/Kamal_final_ver2.xml' #xml file (assumes this is in the same folder as this file)
simend = 20 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


sim_time = []
# end effector position
ee_x = []
ee_y = []
ee_z = []
# control input
inp = []

center = np.array([0.0, -0.03, 0.8])
radius = 0.4
num_points = 100
reference_trajectory = np.array([[center[0] + radius * np.cos(theta), center[2] + radius * np.sin(theta)] 
                                 for theta in np.linspace(0, 2 * np.pi, num_points)]).T

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    
    num_cables = 3  # Update to use all 3 cables
    horizon = 10
    dt = 0.01
    
    current_state = np.array([data.sensordata[3], data.sensordata[5]])

    # Define optimization variables
    x = ca.SX.sym('x', 2, horizon + 1)  # States for position (x, y)
    u = ca.SX.sym('u', num_cables, horizon)  # Control inputs for cable lengths (3 control inputs)

    # Define the cost function
    cost = 0
    constraints = []

    # Initial state constraint
    constraints += [x[:, 0] - current_state]

    # Cable length limits
    # length_min, length_max = model.actuator_ctrlrange.T
    length_min = np.array([-1,-1,-1])
    length_max = np.array([1,0,0])

    for t in range(horizon):
        desired_state = reference_trajectory[:, t]
        cost += ca.sumsqr(x[:, t] - desired_state)  # Tracking cost
        cost += 0.1 * ca.norm_1(u[:, t])  # Penalize control effort

        # Kinematics constraint (simplified for this implementation)
        # Here, we assume the control inputs u[:, t] influence the movement in x and y directions
        data.ctrl[0] = u[0,t]
        data.ctrl[1] = u[1,t]
        data.ctrl[2] = u[2,t]
        mj.mj_forward(model, data)
        next_position = np.array([data.sensordata[3], data.sensordata[5]])
        constraints += [x[:, t + 1] - next_position]

        # Control input limits for each cable
        for i in range(num_cables):
            constraints += [u[i, t] - length_min[i]]
            constraints += [length_max[i] - u[i, t]]

    # Define the NLP problem
    nlp = {'x': ca.vertcat(ca.vec(x), ca.vec(u)), 'f': cost, 'g': ca.vertcat(*constraints)}

    solver_opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 500,
        'ipopt.linear_solver': 'ma27',  # Change linear solver to something more efficient for small problems
        'ipopt.tol': 1e-3  # Set tolerance slightly higher to converge faster with less precision
    }

    # Create an NLP solver
    solver = ca.nlpsol('solver', 'ipopt', nlp,solver_opts)

    # Initial guess
    x0 = np.zeros((2 * (horizon + 1) + num_cables * horizon, 1))

    # Solve the NLP
    sol = solver(x0=x0, lbg=0, ubg=0)

    # Extract the control inputs
    u_opt = sol['x'][-num_cables * horizon:]
    u_opt = np.reshape(u_opt, (num_cables, horizon))

    data.ctrl[0] = u_opt[0,0]  # Update control inputs for all 3 cables
    data.ctrl[1] = u_opt[1,0]
    data.ctrl[2] = u_opt[2,0]

    ee_x.append(data.sensordata[3])
    ee_y.append(data.sensordata[4])
    ee_z.append(data.sensordata[5])
    inp.append(u_opt[:,0])
    sim_time.append(data.time)

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
cam.azimuth = 90
cam.elevation = -30
cam.distance = 5
cam.lookat = np.array([0.0, 1.5, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        plt.figure()
        plt.title('End Effector Position')
        plt.plot(ee_x,ee_z)

        plt.figure()
        plt.title('Control Input')
        plt.plot(sim_time,inp)

        plt.show()
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


