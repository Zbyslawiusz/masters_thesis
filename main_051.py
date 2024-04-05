import math
import time

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


class Simulation():
    def __init__(self, genetic_solution, ui_flag, number_of_links, target_xcor):

        self.x_cor = target_xcor  # x Coordinate that the ball is supposed to hit
        self.control_values = genetic_solution  # ANGLE 1, MOMENTUM 1, ANGLE 2, MOMENTUM 2, ... for all links
        self.draw_ui = ui_flag
        self.number_of_links = number_of_links
        self.first_link_length = 150
        self.firs_link_width = 20
        self.first_link_mass = 1
        self.first_link_x_cor = 600
        self.reduction = 0.95
        self.stand_width = 800
        self.error_sum = []  # Distance from where the ball hit the ground to the intended x coordinate
        self.simulation(self.control_values, self.draw_ui)
        return

    def collision(self, arbiter, space, data):
        """Callback function for the collision handler"""
        print("COLLISION")
        return True

    def ball_with_trail(self, arbiter, space, data):
        return False

    def link_with_trail(self, arbiter, space, data):
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

        # Creating ground
        ground = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground.position = (WIDTH / 2, (HEIGHT - GROUND_THICKNESS / 2))

        ground_shape = pymunk.Poly.create_box(ground, (60_000, GROUND_THICKNESS))
        ground_shape.friction = 1.0
        ground_shape.collision_type = GROUND_COLLISION_TYPE
        space.add(ground, ground_shape)

        # Creating an obstacle
        obstacle_height = 600
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
        manipulator = Manipulator(num_of_links=self.number_of_links + 1,
                                  mass=self.first_link_mass,
                                  reduction=self.reduction,
                                  x_cor=self.first_link_x_cor,
                                  y_cor=(HEIGHT - GROUND_THICKNESS / 2) - GROUND_THICKNESS / 2 -
                                        (self.first_link_length / 2),
                                  length=self.first_link_length,
                                  width=self.firs_link_width,
                                  space=space,
                                  ground=ground,
                                  ball_colltype=BALL_COLLISION_TYPE,
                                  link_colltype=LINK_COLLISION_TYPE,
                                  )

        # manipulator.vertical_manipulator_creator()
        manipulator.horizontal_manipulator_creator()
        manipulator.horizontal_gripper_creator()

        # Desired frame rate and simulation speed
        fps = 200
        dt = 1.0 / fps

        elapsed_time = 0
        for link in manipulator.links[1:]:
            link["angle"] = -pi/2  # Subtracting pi/2 in case of horizontal manipulator creator
        # self.x_cor = 2500  # x Coordinate that the ball is supposed to hit
        registered_distance = 20000

        work_sum = 0

        hit_ground = False
        hit_obstacle = False

        step = 0
        finished = False

        # Main loop
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # print("\nTHE END")
                    return

            # State vector reading (current angles, current angular velocities of links)
            ball_xcor = manipulator.ball.position[0]
            manipulator.update_links()  # Update current and previous angle of every link

            # Detecting collisions with ground and resting point and others
            if not hit_ground:
                handler_ball.begin = manipulator.ball_hit_ground  # Collision between ball and ground
                handler_ball_trail.begin = self.ball_with_trail  # Collision between ball and trail
            # if not is_reversed:
                # handler_link.begin = manipulator.link_reversed  # Collision between links and stand
            handler_manipulator_trail.begin = self.link_with_trail  # Collision between links and trail
            if not hit_obstacle:  # !!!!!!!!!!!!!!!!Tu moze byc blad - sprawdzanie kolizji z przeszkoda tylko raz
                handler_obstacle.begin = manipulator.obstacle_hit  # Collision between ball and obstacle
                hit_obstacle = True

            if manipulator.ball_hit_the_ground and not hit_ground:
                registered_distance = abs(ball_xcor - self.x_cor)
                # print(registered_distance)
                hit_ground = True

            # print(f"Ball's x, y coordinates: {manipulator.ball.position[0]}, {manipulator.ball.position[1]}\n"
            #       f"Ball's vx, vy velocity: {manipulator.ball.velocity[0]}, {manipulator.ball.velocity[1]}\n")

            # Moving the links
            i = 0
            for link in manipulator.links[1:]:
                traversed_angle = abs(link["angle"] - link["previous_angle"])
                if self.control_values[i + 1] >= 20:
                    force = 10 * -1000 * math.sin(link["angle"])
                else:
                    force = self.control_values[i + 1] * -1000 * math.sin(link["angle"])  # Used to create momentum
                desired_angle = self.control_values[i]
                work_sum += abs(force * traversed_angle)
                if link["angle"] < desired_angle and not link["is set"]:
                    manipulator.simple_throw(force=force, link=link)
                else:
                    link["is set"] = True
                manipulator.pid_brake(link=link)

                i += 2

            if ui_flag:
                # Clear screen
                screen.fill(pygame.Color("grey"))
                # Draw stuff
                space.debug_draw(draw_options)
                pygame.display.flip()
                clock.tick(fps)
                # print(link_ang_vel)

                if step < 10 and not hit_ground and not finished:
                    step += 1
                    if step == 1:
                        pos = manipulator.ball.position
                elif step >= 10 and not hit_ground and not finished:
                    # Creating the trail
                    trail = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    trail.position = pos
                    trail_shape = pymunk.Circle(trail, radius=3)
                    trail_shape.collision_type = 37

                    space.add(trail, trail_shape)
                    step = 0

            # Update physics
            if not finished:
                space.step(dt)
                elapsed_time += dt
            else:
                time.sleep(3)
                return  # Close pymunk in 3 seconds after finishing the simulation with GUI on
            if hit_ground:
                manipulator.ball.velocity = (0, 0)

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
