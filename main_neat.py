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

from manipulator_02 import Manipulator

WIDTH, HEIGHT = 1500, 1000

LINK_COLLISION_TYPE = 0
BALL_COLLISION_TYPE = 10
GROUND_COLLISION_TYPE = 1
REST_COLLISION_TYPE = 2
OBSTACLE_COLLISION_TYPE = 3
GROUND_THICKNESS = 50


class Simulation:
    def __init__(self, net, ui_flag, number_of_links, target_xcor, interpolation=4,
                 time_of_throw=1_000_000, picks_or_not="False", gripper="stiff", type="best"):

        self.filenames = []  # List storing filenames of screenshots if they're taken

        self.max_force = 0.8  # Simulates physical constraints and safety limits of manipulator's servomotors

        self.x_cor = target_xcor  # x Coordinate that the ball is supposed to hit
        self.net = net  # ANGLE 1, MOMENTUM 1, ANGLE 2, MOMENTUM 2, ... for all links
        self.draw_ui = ui_flag  # for 1st link ANGLE 1, TIMESTAMP 1, ,,, ANGLE n, TIMESTAMP n, for all links

        # To make screenshots or not
        self.make_pics = False
        if picks_or_not:
            # self.make_pics = True
            self.interval = time_of_throw / 16

            # Folder containing screenshots is cleaned every time new screenshots are to be taken
            if type == "best":
                folder = "./Pymunk_pics/Best_sim"
            elif type == "acceptable":
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
        self.gripper_type = gripper

        self.first_link_length = 150
        self.firs_link_width = 20
        self.first_link_mass = 1
        self.first_link_x_cor = 600
        self.reduction = 0.95
        self.stand_width = 800
        self.error_sum = []  # Distance from where the ball hit the ground to the intended x coordinate
        self.simulation(self.net, self.draw_ui)

        # Moving the screenshot files to their designated folder
        if self.draw_ui and type == "acceptable":
            for filename in self.filenames:
                shutil.move(f"./{filename}", "./Pymunk_pics/Acceptable_sim")
        elif self.draw_ui and type == "best":
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
        ground = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground.position = (WIDTH / 2, (HEIGHT - GROUND_THICKNESS / 2))

        ground_shape = pymunk.Poly.create_box(ground, (60_000, GROUND_THICKNESS))
        ground_shape.friction = 1.0
        ground_shape.collision_type = GROUND_COLLISION_TYPE
        space.add(ground, ground_shape)

        # Creating an obstacle
        obstacle_height = 20
        obstacle_width = 10
        obstacle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        obstacle.position = (self.first_link_x_cor + 800, HEIGHT - GROUND_THICKNESS / 2 - GROUND_THICKNESS / 2 -
                             obstacle_height / 2)

        obstacle_shape = pymunk.Poly.create_box(obstacle, (obstacle_width, obstacle_height))
        obstacle_shape.friction = 0.5
        obstacle_shape.collision_type = OBSTACLE_COLLISION_TYPE
        space.add(obstacle, obstacle_shape)

        # Creating resting point for the manipulator
        stand = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
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
        registered_distance = 2_000_000  # Default value of distance, penalizes manipulator not doing anything

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
            ball_ycor = manipulator.ball.position[1]
            ball_xvel = manipulator.ball.velocity[0]
            ball_yvel = manipulator.ball.velocity[1]
            manipulator.update_links()  # Update current and previous angle of every link ------------------------------

            # Detecting collisions with ground and resting point and others --------------------------------------------
            if not hit_ground:
                handler_ball.begin = manipulator.ball_hit_ground  # Collision between ball and ground
                handler_ball_trail.begin = self.ball_with_trail  # Collision between ball and trail
            # if not is_reversed:
                # handler_link.begin = manipulator.link_reversed  # Collision between links and stand
            handler_manipulator_trail.begin = self.link_with_trail  # Collision between links and trail
            if not hit_obstacle:  # !!!!!!!!!!!!!!!!Tu moze byc blad - sprawdzanie kolizji z przeszkoda tylko raz
                handler_obstacle.begin = manipulator.obstacle_hit  # Collision between ball and obstacle
                hit_obstacle = True
            if not ball_released and elapsed_time > 0.2:  # Checks if the ball stopped touching the gripper
                handler_ball_release.separate = manipulator.ball_not_touching_gripper
                ball_released = True

            if manipulator.ball_hit_the_ground and not hit_ground:
                registered_distance = abs(ball_xcor - self.x_cor)
                # print(registered_distance)
                hit_ground = True

            # print(f"Ball's x, y coordinates: {manipulator.ball.position[0]}, {manipulator.ball.position[1]}\n"
            #       f"Ball's vx, vy velocity: {manipulator.ball.velocity[0]}, {manipulator.ball.velocity[1]}\n")

            # ----------------------------------------------------------------------------------------------------------
            # Activating the neural network based on current error every main loop iteration ---------------------------
            # ----------------------------------------------------------------------------------------------------------
            # Calculating error sum used for first net activation
            current_angles = [link["angle"] for link in manipulator.links[1:]]
            previous_angles = [link["previous_angle"] for link in manipulator.links[1:]]
            error = [ball_xcor, ball_ycor, ball_xvel, ball_yvel] + current_angles + previous_angles

            solution = self.net.activate(error)  # Acquiring solution from the neural network
            # print(solution)
            # ----------------------------------------------------------------------------------------------------------
            # End of NEAT algorithm part of code -----------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            # for 1st link ANGLE 1, ... ANGLE n for all links

            # Moving the links -----------------------------------------------------------------------------------------
            i = 0
            for link in manipulator.links[1:]:
                traversed_angle = abs(link["angle"] - link["previous_angle"])
                # Passing the current timestamp to interpolated function in order to calculate current desired angle
                # if not ball_released:
                desired_angle = solution[i]
                link["desired angle"] = desired_angle
                # Correcting the angles for pymunk
                if desired_angle > 2 * pi:
                    desired_angle = 0 + desired_angle % (2 * pi)
                elif desired_angle < -2 * pi:
                    desired_angle = 0 - desired_angle % (2 * pi)
                else:
                    pass
                # else:
                #     desired_angle = link["desired angle"]
                #     # print(f"Desired angle: {desired_angle}, elapsed time: {elapsed_time}")
                if i == 0:
                    angle_error = desired_angle - link["angle"]
                else:
                    angle_error = desired_angle - manipulator.links[i - 1]["angle"]
                # print(f"Error: {error}")
                force = manipulator.pid_force_calculator(error=angle_error, dt=dt)
                if force > self.max_force:
                    force = self.max_force
                elif force < -self.max_force:
                    force = -self.max_force
                work_sum += abs(force * traversed_angle)
                manipulator.simple_throw(force=force*10000, link=link)  # Moving the link

                if self.gripper_type == "robotic" and elapsed_time >= solution[-1] and not open_gripper:
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
                    trail = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    trail.position = pos
                    trail_shape = pymunk.Circle(trail, radius=3)
                    trail_shape.collision_type = 37

                    space.add(trail, trail_shape)
                    step = 0

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
                    return self.error_sum
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
                    return self.error_sum
                # return self.error_sum

        # if __name__ == "__simulation__":
        #     sys.exit(simulation(machine_input))
