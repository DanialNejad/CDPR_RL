import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import os
import csv

xml_path = 'Kamal_final_ver2.xml' #xml file (assumes this is in the same folder as this file)
simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


t = []
# end effector position
ee_x = []
ee_y = []
ee_z = []
# end effector velocity
ee_vx = []
ee_vy = []
ee_vz = []
# cable length
l1 = []
l2 = []
l3 = []
# upper cable force
fu1 = []
fu2 = []
fu3 = []
# left cable force
fl1 = []
fl2 = []
fl3 = []
# right cable force
fr1 = []
fr2 = []
fr3 = []

inp = []


def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    t.append(data.time)

    l1.append(data.sensordata[0])
    l2.append(data.sensordata[1])
    l3.append(data.sensordata[2])

    ee_x.append(data.sensordata[3])
    ee_y.append(data.sensordata[4])
    ee_z.append(data.sensordata[5])

    ee_vx.append(data.sensordata[6])
    ee_vy.append(data.sensordata[7])
    ee_vz.append(data.sensordata[8])

    # fu1.append(data.sensordata[3])
    # fu2.append(data.sensordata[4])
    # fu3.append(data.sensordata[5])

    # fl1.append(data.sensordata[6])
    # fl2.append(data.sensordata[7])
    # fl3.append(data.sensordata[8])

    data.ctrl[0] = 0.5*np.cos(0.3*np.pi*data.time)
    # data.ctrl[0] = 100000*(np.heaviside(data.time - 2,0))

    inp.append(data.ctrl[0])
    data.ctrl[1] = 0.5*np.sin(0.3*np.pi*data.time) - 0.5

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
        # File path to save CSV
        # csv_file = 'data.csv'

        # Combine arrays into a list
        # theta = np.degrees(theta) + 45
        # l1 = l1 + np.radians(45)
        # theta_dot = np.degrees(theta_dot)
        # data = [t, theta, theta_dot]
        print(len(data.sensordata))
        # print(ee_y)
        # data_rearranged = np.column_stack(data)

        # Save rearranged data to CSV
        # np.savetxt(csv_file, data_rearranged, delimiter=',')
        # print(theta)
        # print(theta_dot)

        plt.figure()
        plt.title('End Effector Position')
        plt.plot(ee_x,ee_z)

        plt.figure()
        plt.title('End Effector Velocity')
        plt.subplot(3,1,1)
        plt.plot(t,ee_vx)
        plt.ylabel('$V_x$')
        plt.subplot(3,1,2)
        plt.plot(t,ee_vy)
        plt.ylabel('$V_y$')
        plt.subplot(3,1,3)
        plt.plot(t,ee_vz)
        plt.ylabel('$V_z$')
        plt.xlabel('Time (Sec)')

        # plt.figure()
        # plt.plot(t,inp)
        # plt.xlabel('Time (Sec)')
        # plt.ylabel('Force')

        plt.figure()
        plt.title('Cable Length')
        plt.subplot(3,1,1)
        plt.plot(t,l1)
        plt.ylabel('Upper Cable Length')
        plt.subplot(3,1,2)
        plt.plot(t,l2)
        plt.ylabel('Left Cable Length')
        plt.subplot(3,1,3)
        plt.plot(t,l3)
        plt.ylabel('Right Cable Length')
        plt.xlabel('Time (Sec)')

        # plt.figure()
        # plt.title('Upper Cable Force')
        # plt.subplot(3,1,1)
        # plt.plot(t,fu1)
        # plt.ylabel('$F_x$')
        # plt.subplot(3,1,2)
        # plt.plot(t,fu2)
        # plt.ylabel('$F_y$')
        # plt.subplot(3,1,3)
        # plt.plot(t,fu3)
        # plt.ylabel('$F_z$')
        # plt.xlabel('Time (Sec)')

        # plt.figure()
        # plt.title('Left Cable Force')
        # plt.subplot(3,1,1)
        # plt.plot(t,fl1)
        # plt.ylabel('$F_x$')
        # plt.subplot(3,1,2)
        # plt.plot(t,fl2)
        # plt.ylabel('$F_y$')
        # plt.subplot(3,1,3)
        # plt.plot(t,fl3)
        # plt.ylabel('$F_z$')
        # plt.xlabel('Time (Sec)')

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


