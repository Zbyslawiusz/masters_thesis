import time
import uuid
import os, shutil
from PIL import ImageGrab
import pygetwindow as gw

import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
from math import pi

import numpy as np
from scipy.interpolate import interp1d

from manipulator_02 import Manipulator

WIDTH, HEIGHT = 1500, 1000

LINK_COLLISION_TYPE = 0
BALL_COLLISION_TYPE = 10
GROUND_COLLISION_TYPE = 1
REST_COLLISION_TYPE = 2
OBSTACLE_COLLISION_TYPE = 3
GROUND_THICKNESS = 50


class Simulation:
    def __init__(self, genetic_solution, ui_flag, number_of_links, target_xcor, interpolation,
                 time_of_throw=1_000_000, picks_or_not=False, gripper="stiff", sim_type="best", throw_type="target"):

        self.filenames = []  # List storing filenames of screenshots if they're taken
        self.throw_type = throw_type

        self.max_force = 0.8  # Simulates physical constraints and safety limits of manipulator's servomotors

        self.x_cor = target_xcor  # x Coordinate that the ball is supposed to hit
        self.control_values = genetic_solution  # ANGLE 1, MOMENTUM 1, ANGLE 2, MOMENTUM 2, ... for all links
        self.draw_ui = ui_flag  # for 1st link ANGLE 1, TIMESTAMP 1, ,,, ANGLE n, TIMESTAMP n, for all links

        # To make screenshots or not
        if picks_or_not:
            self.make_pics = True
            print("make_picks = True")
            self.interval = time_of_throw / 16

            # Folder containing screenshots is cleaned every time new screenshots are to be taken
            if sim_type == "best":
                folder = "./Pymunk_pics/Best_sim"
            elif sim_type == "acceptable":
                folder = "./Pymunk_pics/Acceptable_sim"
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            self.make_pics = False
            self.interval = 0

        self.number_of_links = number_of_links
        self.interpolation = interpolation
        self.interp_functions = []  # This list contains interpolated functions for individual links
        self.gripper_type = gripper

        # for 1st link ANGLE 1, TIMESTAMP 1, ,,, ANGLE n, TIMESTAMP n, for all links
        for i in range(0, self.number_of_links):
            angles = []  # Contains desired angles for each link
            timestamps = []  # Stores timestamps corresponding to desired angles for each link
            values = self.control_values[i*self.interpolation*2:i*self.interpolation*2+self.interpolation*2]
            j = 0
            for _ in range(0, len(values)):
                if _ % 2 != 0:  # Check every even number in genetic solution to get angle, odd numbers store timestamps
                    if j == 0:
                        timestamps.append(0)  # Timestamps start at time = 0
                        j += 1
                    else:
                        # They have to be monotonically increasing
                        timestamps.append(timestamps[j - 1] + abs(values[_]))
                        j += 1

                else:  # Even numbers store angles
                    angles.append(values[_])

            if len(timestamps) < 4:
                timestamps.insert(0, 0)
                angles.insert(0, 0)

            for _ in range(0, len(timestamps)):
                # Timestamps have to be monotonically increasing, if they're not scipy will crash
                try:
                    if timestamps[_+1] <= timestamps[_]:
                        timestamps[_+1] = timestamps[_] + 0.001
                except IndexError:
                    pass
                if angles[_] < -pi/2:  # Making sure angles are in a correct range
                    angles[_] = -pi/2
                elif angles[_] > pi/2:
                    angles[_] = pi/2
            # Simulation starts in time=0, therefore interpolation must include value range starting at 0 or scipy will crash
            # if timestamps[0] > 0 or timestamps[0] < 0:
            #     timestamps[0] = 0
                # timestamps.insert(0, 0)
                # angles.insert(0, 0)

            # It is now possible to interpolate values separated into angles and timestamps
            # print(f"np.array(timestamps): {np.array(timestamps)}, np.array(angles): {np.array(angles)}")
            interp_func = interp1d(np.array(timestamps), np.array(angles), kind="cubic", fill_value="extrapolate")
            self.interp_functions.append(interp_func)

        self.first_link_length = 150
        self.firs_link_width = 20
        self.first_link_mass = 1
        self.first_link_x_cor = 600
        self.reduction = 0.95
        self.stand_width = 800
        self.error_sum = []  # Distance from where the ball hit the ground to the intended x coordinate
        self.simulation(self.control_values, self.draw_ui)

        # Moving the screenshot files to their designated folder
        if self.draw_ui and sim_type == "acceptable":
            for filename in self.filenames:
                shutil.move(f"./{filename}", "./Pymunk_pics/Acceptable_sim")
        elif self.draw_ui and sim_type == "best":
            for filename in self.filenames:
                shutil.move(f"./{filename}", "./Pymunk_pics/Best_sim")
        return

    def collision(self, arbiter, space, data):
        """Callback function for the collision handler"""
        print("COLLISION")
        return True

    def ball_with_trail(self, arbiter, space, data):
        """Callback function for the collision handler"""
        return False

    def link_with_trail(self, arbiter, space, data):
        """Callback function for the collision handler"""
        return False

    def simulation(self, genetic_solution, ui_flag):
        """Function for simulating manipulator throwing a ball"""

        # Physics engine initialization
        pygame.init()

        # Creating a screen object
        if ui_flag:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            draw_options = pymunk.pygame_util.DrawOptions(screen)
            clock = pygame.time.Clock()

            # print(gw.getAllTitles())  # 'pygame window'

        running = True
        # font = pygame.font.SysFont("Arial", 16)

        # Initializing instance of space
        space = pymunk.Space(threaded=True)
        space.threads = 10
        space.gravity = Vec2d(0.0, 900.0)

        # Creating collision handlers
        handler_ball = space.add_collision_handler(BALL_COLLISION_TYPE, GROUND_COLLISION_TYPE)
        # handler_link = space.add_collision_handler(LINK_COLLISION_TYPE, REST_COLLISION_TYPE)
        handler_obstacle = space.add_collision_handler(BALL_COLLISION_TYPE, OBSTACLE_COLLISION_TYPE)
        handler_ball_trail = space.add_collision_handler(BALL_COLLISION_TYPE, 37)
        handler_manipulator_trail = space.add_collision_handler(LINK_COLLISION_TYPE, 37)
        handler_ball_release = space.add_collision_handler(BALL_COLLISION_TYPE, LINK_COLLISION_TYPE)

        # Creating ground
        ground = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground.position = (WIDTH / 2, (HEIGHT - GROUND_THICKNESS / 2))

        ground_shape = pymunk.Poly.create_box(ground, (60_000, GROUND_THICKNESS))
        ground_shape.friction = 1.0
        ground_shape.collision_type = GROUND_COLLISION_TYPE
        space.add(ground, ground_shape)

        # Creating an obstacle for 'target' type of throw
        if self.throw_type == "target":
            obstacle_height = 400
            obstacle_width = 10
            obstacle = pymunk.Body(body_type=pymunk.Body.STATIC)
            obstacle.position = (self.first_link_x_cor + 800, HEIGHT - GROUND_THICKNESS / 2 - GROUND_THICKNESS / 2 -
                                 obstacle_height / 2)

            obstacle_shape = pymunk.Poly.create_box(obstacle, (obstacle_width, obstacle_height))
            obstacle_shape.friction = 0.5
            obstacle_shape.collision_type = OBSTACLE_COLLISION_TYPE
            space.add(obstacle, obstacle_shape)

        if self.throw_type == "gimmick":
            offset = 800
            width = 250
            gap = 40
            self.first_link_x_cor = 400

            obst1_h = 600
            obst1_w = 10
            obst1 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst1.position = (self.first_link_x_cor + offset, HEIGHT - GROUND_THICKNESS / 2 - GROUND_THICKNESS / 2 -
                              obst1_h / 2)
            obst1_shape = pymunk.Poly.create_box(obst1, (obst1_w, obst1_h))
            obst1_shape.collision_type = OBSTACLE_COLLISION_TYPE
            space.add(obst1, obst1_shape)

            obst2_h = 300
            obst2_w = 10
            obst2 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst2.position = (self.first_link_x_cor + offset, HEIGHT - GROUND_THICKNESS / 2 - GROUND_THICKNESS / 2 -
                              obst2_h / 2 - obst1_h - gap)
            obst2_shape = pymunk.Poly.create_box(obst2, (obst2_w, obst2_h))
            obst2_shape.collision_type = OBSTACLE_COLLISION_TYPE
            space.add(obst2, obst2_shape)

            obst3_h = 650
            obst3_w = 10
            obst3 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst3.position = (self.first_link_x_cor + offset + width, HEIGHT - GROUND_THICKNESS / 2 -
                              GROUND_THICKNESS / 2 - obst3_h / 2)
            obst3_shape = pymunk.Poly.create_box(obst3, (obst3_w, obst3_h))
            obst3_shape.collision_type = 256
            space.add(obst3, obst3_shape)

            # dynamic obstacle -----------------------------------------------------------------------------------------
            obst4 = pymunk.Body(100, pymunk.moment_for_box(mass=100, size=(200, 10)))
            obst4.position = (self.first_link_x_cor + offset + width/2, HEIGHT - GROUND_THICKNESS / 2 -
                              GROUND_THICKNESS / 2 - obst1_h - gap / 2)
            obst4_shape = pymunk.Poly.create_box(body=obst4, size=(200, 10))
            obst4_shape.friction = 1.0

            p = Vec2d(self.first_link_x_cor + offset + width/2, HEIGHT - GROUND_THICKNESS / 2 - GROUND_THICKNESS / 2 -
                      obst1_h - gap / 2)
            b0 = space.static_body

            # anchor joint
            anchor_joint = pymunk.PivotJoint(b0, obst4, p)
            anchor_joint.error_bias = 0
            anchor_motor = pymunk.SimpleMotor(a=b0, b=obst4, rate=5)  # oryginalnie rate=2
            anchor_motor.max_force = 1000000  # oryginalnie 100000

            space.add(obst4, obst4_shape, anchor_joint, anchor_motor)
            # dynamic obstacle -----------------------------------------------------------------------------------------

            obst5_h = 10
            obst5_w = 150
            obst5 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst5.position = (self.first_link_x_cor + offset + obst1_w/2 + obst5_w/2,
                              HEIGHT - GROUND_THICKNESS - obst1_h - gap - 100)
            obst5_shape = pymunk.Poly.create_box(obst5, (obst5_w, obst5_h))
            obst5_shape.collision_type = 256
            space.add(obst5, obst5_shape)

            obst6_h = 500
            obst6_w = 10
            obst6 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst6.position = (self.first_link_x_cor + offset + width/2, HEIGHT - GROUND_THICKNESS - obst6_h/2)
            obst6_shape = pymunk.Poly.create_box(obst6, (obst6_w, obst6_h))
            obst6_shape.collision_type = 256
            space.add(obst6, obst6_shape)

            obst7_h = 515
            obst7_w = 10
            obst7 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst7.position = (self.first_link_x_cor + offset + width/4, HEIGHT - GROUND_THICKNESS - obst7_h/2)
            obst7_shape = pymunk.Poly.create_box(obst7, (obst7_w, obst7_h))
            obst7_shape.collision_type = 256
            space.add(obst7, obst7_shape)

            obst8_h = 200
            obst8_w = 10
            obst8 = pymunk.Body(body_type=pymunk.Body.STATIC)
            obst8.position = (self.first_link_x_cor + offset + 3*width/4, HEIGHT - GROUND_THICKNESS - obst8_h / 2)
            obst8_shape = pymunk.Poly.create_box(obst8, (obst8_w, obst8_h))
            obst8_shape.collision_type = 256
            space.add(obst8, obst8_shape)

            # Setting the correct target x_cor for the ball to hit
            self.x_cor = obst3.position[0] - (obst3.position[0] - obst8.position[0]) / 2

        # Creating resting point for the manipulator
        stand = pymunk.Body(body_type=pymunk.Body.STATIC)
        stand.position = (self.first_link_x_cor - 50 - self.stand_width / 2,
                          HEIGHT - GROUND_THICKNESS - self.first_link_length / 2 + self.firs_link_width / 2 - 4)

        stand_shape = pymunk.Poly.create_box(stand, (self.stand_width,
                                                     self.first_link_length - self.firs_link_width / 2))
        stand_shape.friction = 0.5
        stand_shape.collision_type = REST_COLLISION_TYPE
        space.add(stand, stand_shape)

        # Creating manipulator object
        manipulator = Manipulator(
            num_of_links=self.number_of_links + 1,
            mass=self.first_link_mass,
            reduction=self.reduction,
            x_cor=self.first_link_x_cor,
            y_cor=(HEIGHT - GROUND_THICKNESS / 2) - GROUND_THICKNESS / 2 - (self.first_link_length / 2),
            length=self.first_link_length,
            width=self.firs_link_width,
            space=space,
            ground=ground,
            ball_colltype=BALL_COLLISION_TYPE,
            link_colltype=LINK_COLLISION_TYPE,
        )

        # manipulator.vertical_manipulator_creator()
        if self.gripper_type == "stiff":
            manipulator.horizontal_manipulator_creator_stiff_grabber()
        elif self.gripper_type == "robotic":
            manipulator.horizontal_manipulator_creator_robotic_grabber()
        else:
            return "Error, wrong gripper type"
        # manipulator.horizontal_gripper_creator()

        # Desired frame rate and simulation speed
        fps = 200
        dt = 1.0 / fps

        for link in manipulator.links[1:]:
            link["angle"] = -pi/2  # Subtracting pi/2 in case of horizontal manipulator creator
        # self.x_cor = 2500  # x Coordinate that the ball is supposed to hit
        if self.throw_type == "target" or self.throw_type == "gimmick":
            registered_distance = 2_000_000  # Default value of distance, penalizes manipulator not doing anything
        elif self.throw_type == "far":
            registered_distance = -2_000_000

        work_sum = 0

        hit_ground = False
        hit_obstacle = False
        open_gripper = False
        ball_released = False

        step = 0
        finished = False

        elapsed_time = 0
        pick_time = self.interval
        # Main loop ----------------------------------------------------------------------------------------------------
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # print("\nTHE END")
                    return

            # State vector reading (current angles, current angular velocities of links) -------------------------------
            ball_xcor = manipulator.ball.position[0]
            manipulator.update_links()  # Update current and previous angle of every link ------------------------------

            # Detecting collisions with ground and resting point and others --------------------------------------------
            if not hit_ground:
                handler_ball.begin = manipulator.ball_hit_ground  # Collision between ball and ground
                # handler_ball_trail.begin = self.ball_with_trail  # Collision between ball and trail
            # if not is_reversed:
                # handler_link.begin = manipulator.link_reversed  # Collision between links and stand
            # handler_manipulator_trail.begin = self.link_with_trail  # Collision between links and trail
            if not hit_obstacle:  # !!!!!!!!!!!!!!!!Tu moze byc blad - sprawdzanie kolizji z przeszkoda tylko raz
                handler_obstacle.begin = manipulator.obstacle_hit  # Collision between ball and obstacle
                hit_obstacle = True
            if not ball_released and elapsed_time > 0.2:  # Checks if the ball stopped touching the gripper
                handler_ball_release.separate = manipulator.ball_not_touching_gripper
                ball_released = True

            if manipulator.ball_hit_the_ground and not hit_ground:
                if self.throw_type == "target" or self.throw_type == "gimmick":
                    registered_distance = abs(ball_xcor - self.x_cor)
                elif self.throw_type == "far":
                    registered_distance = ball_xcor
                # print(registered_distance)
                hit_ground = True

            # print(f"Ball's x, y coordinates: {manipulator.ball.position[0]}, {manipulator.ball.position[1]}\n"
            #       f"Ball's vx, vy velocity: {manipulator.ball.velocity[0]}, {manipulator.ball.velocity[1]}\n")

            # Moving the links -----------------------------------------------------------------------------------------
            i = 0
            for link in manipulator.links[1:]:
                traversed_angle = abs(link["angle"] - link["previous_angle"])
                # Passing the current timestamp to interpolated function in order to calculate current desired angle
                if not ball_released:
                    desired_angle = self.interp_functions[i](elapsed_time)
                    link["desired angle"] = desired_angle
                    # Correcting the angles for pymunk
                    if desired_angle > 2 * pi:
                        desired_angle = 0 + desired_angle % (2 * pi)
                    elif desired_angle < -2 * pi:
                        desired_angle = 0 - desired_angle % (2 * pi)
                    else:
                        pass
                else:
                    desired_angle = link["desired angle"]
                    # print(f"Desired angle: {desired_angle}, elapsed time: {elapsed_time}")
                if i == 0:
                    error = desired_angle - link["angle"]
                else:
                    error = desired_angle - manipulator.links[i - 1]["angle"]
                # print(f"Error: {error}")
                force = manipulator.pid_force_calculator(error=error, dt=dt)
                if force > self.max_force:
                    force = self.max_force
                elif force < -self.max_force:
                    force = -self.max_force
                work_sum += abs(force * traversed_angle)
                manipulator.simple_throw(force=force*10000, link=link)  # Moving the link

                # The last value of the solution is a timestamp used to open the gripper claws
                if self.gripper_type == "robotic" and elapsed_time >= self.control_values[-1] and not open_gripper:
                # if self.gripper_type == "robotic" and elapsed_time >= 1 and not open_gripper:
                    manipulator.right_claw_motor.rate *= -10  # Reverse robotic claw motors and open the gripper
                    manipulator.left_claw_motor.rate *= -10
                    open_gripper = True
                i += 1

            # Drawing pymunk UI ----------------------------------------------------------------------------------------
            if ui_flag:
                # Clear screen
                screen.fill(pygame.Color("grey"))
                # Draw stuff
                space.debug_draw(draw_options)
                pygame.display.flip()
                clock.tick(fps * 0.25)  # For slow motion
                # print(link_ang_vel)

                # Making screenshots of the pymunk window
                if self.make_pics and elapsed_time >= pick_time:
                    print("\n--------------------------------------------------------------------------------------\n"
                          "Screenshot Done"
                          "\n--------------------------------------------------------------------------------------\n")
                    window = gw.getWindowsWithTitle('pygame window')[0]
                    # Capturing a specific region of the screen (left, top, right, bottom)
                    screenshot = ImageGrab.grab(bbox=(window.left,
                                                      window.top,
                                                      window.bottomright[0],
                                                      window.bottomright[1]))
                    # Saving the screenshot as a file with a unique name
                    filename = f"{uuid.uuid4().hex}.png"
                    screenshot.save(filename)
                    # Closing the screenshot
                    screenshot.close()
                    self.filenames.append(filename)
                    pick_time += self.interval

                # Drawing the trail-------------------------------------------------------------------------------------
                if step < 10 and not hit_ground and not finished:
                    step += 1
                    if step == 1:
                        pos = manipulator.ball.position
                elif step >= 10 and not hit_ground and not finished:
                    # Creating the trail balls
                    trail = pymunk.Body(body_type=pymunk.Body.STATIC)
                    trail.position = pos
                    trail_shape = pymunk.Circle(trail, radius=3)
                    trail_shape.collision_type = 37

                    space.add(trail, trail_shape)
                    step = 0

                handler_ball_trail.begin = self.ball_with_trail  # Collision between ball and trail
                handler_manipulator_trail.begin = self.link_with_trail  # Collision between links and trail

            # Update physics -------------------------------------------------------------------------------------------
            if not finished:
                space.step(dt)
                elapsed_time += dt
            else:
                time.sleep(3)
                return  # Close pymunk in 3 seconds after finishing the simulation with GUI on
            if hit_ground:
                manipulator.ball.velocity = (0, 0)

            # Finishing simulation by hitting the ground ---------------------------------------------------------------
            if hit_ground and not finished:
                finished = True
                self.error_sum = [registered_distance, hit_obstacle, elapsed_time, work_sum]
                if ui_flag:
                    print(f"\nDistance: {registered_distance}.\n"
                          f"Elapsed time: {elapsed_time}.\n"
                          f"Total work: {work_sum}.")
                else:
                    return
                # return self.error_sum

            # Timeout --------------------------------------------------------------------------------------------------
            elif elapsed_time > 7 and not finished:
                finished = True
                self.error_sum = [registered_distance, hit_obstacle, elapsed_time, work_sum]
                if ui_flag:
                    print(f"\nDistance: {registered_distance}.\n"
                          f"Elapsed time: {elapsed_time}.\n"
                          f"Total work: {work_sum}.")
                else:
                    return
                # return self.error_sum

        # if __name__ == "__simulation__":
        #     sys.exit(simulation(machine_input))
