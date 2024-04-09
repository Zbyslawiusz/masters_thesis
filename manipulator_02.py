import pymunk
from math import pi

# COLLISION_TYPE_DEFAULT = 0


class Manipulator:
    def __init__(self, num_of_links, mass, reduction, x_cor, y_cor, length, width, ground, space,
                 ball_colltype, link_colltype):
        self.link_collision_type = link_colltype
        self.num_of_links = num_of_links
        self.first_link_mass = mass
        self.mass_and_length_red = reduction  # How much each consecutive link's mass and dimensions will be reduced
        self.links = []
        self.x = x_cor
        self.y = y_cor
        # Dimensions of the first link, they are reverted if horizontal manipulator creator is used
        self.dimensions = [width, length]
        self.ground = ground
        self.space = space
        self.throw_on = True
        self.ball_radius = 10

        # self.kp = 1
        # self.ki = 0.1
        # self.kd = 0.001
        self.kp = 10
        self.ki = 0.5
        self.kd = 0.09

        self.prev_error = 0
        self.integral = 0

        self.ball_hit_the_ground = False
        self.link_is_reversed = False
        self.obstacle_is_hit = False

        self.ball = pymunk.Body(1, pymunk.moment_for_circle(mass=1, inner_radius=0, outer_radius=10))
        self.ball.position = 540, 0
        self.ball_shape = pymunk.Circle(self.ball, 10)
        self.ball_shape.collision_type = ball_colltype
        self.ball_shape.friction = 1.0
        # self.space.add(self.ball, self.ball_shape)

        first_link = pymunk.Body(mass, pymunk.moment_for_box(mass, (width, length)), body_type=pymunk.Body.STATIC)
        first_link.position = self.x, self.y

        first_link_shape = pymunk.Poly.create_box(body=first_link, size=(width, length), radius=0)
        first_link_shape.collision_type = self.link_collision_type
        # first_link_shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(first_link, first_link_shape)
        self.links.append({
            "link": first_link,
            "angle": 0.0,
            "length": length,
            "position": first_link.position,
            "force": 0,
            "is set": False,
            })
        # self.horizontal_manipulator_creator()

    def pid_force_calculator(self, error, dt):
        """This method calculates momentum that is used to rotate links based on error between current angle and
        desired angle from interpolated GA angles and timestamps"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        force = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return force

    def vertical_manipulator_creator(self):
        """This method creates links of the manipulator. Links are standing upon creation"""
        mass = self.first_link_mass
        width = self.dimensions[0]
        length = self.dimensions[1]
        y = self.y - length/2
        for _ in range(1, self.num_of_links):
            mass *= (self.mass_and_length_red ** 2)
            width *= self.mass_and_length_red
            length *= self.mass_and_length_red

            link_moment = pymunk.moment_for_box(mass, (width, length))
            link = pymunk.Body(mass, link_moment)
            link.position = self.x, y - length/2

            link_shape = pymunk.Poly.create_box(body=link, size=(width, length), radius=0)
            link_shape.collision_type = self.link_collision_type
            link_shape.friction = 1.0
            # link_shape.filter = pymunk.ShapeFilter(group=1)

            link_joint = pymunk.PivotJoint(self.links[_ - 1]["link"],   # a
                                           link,                # b
                                           (self.x, y))    # pivot point
            link_joint.error_bias = 0
            link_joint.collide_bodies = False

            self.links.append({
                "link": link,
                "angle": 0.0,
                "length": length,
                "position": link.position,
                "force": 0,
                "is set": False,
            })
            self.space.add(link, link_shape, link_joint)

            y -= length

    def horizontal_manipulator_creator(self):
        """This method creates links of the manipulator. Links are laying on their left side upon creation"""
        mass = self.first_link_mass
        width = self.dimensions[1]  #
        length = self.dimensions[0]
        y = self.y - width/2
        x = self.x
        for i in range(1, self.num_of_links):
            mass *= self.mass_and_length_red
            width *= self.mass_and_length_red
            # length *= self.mass_and_length_red
            # y -= length/2

            link = pymunk.Body(mass, pymunk.moment_for_box(mass, (width, length)))
            link.position = x - width/2, y

            link_shape = pymunk.Poly.create_box(body=link, size=(width, length), radius=0)
            link_shape.collision_type = self.link_collision_type
            link_shape.friction = 1.0
            # link_shape.filter = pymunk.ShapeFilter(group=1)

            link_joint = pymunk.PivotJoint(self.links[i - 1]["link"],   # a
                                           link,                # b
                                           (x, y))    # pivot point
            link_joint.error_bias = 0
            link_joint.collide_bodies = False

            self.links.append({
                "link": link,
                "angle": 0,
                "previous_angle": 0,
                "length": width,
                "force": 0,
                "is set": False,
            })
            self.space.add(self.links[-1]["link"], link_shape, link_joint)

            x -= width

    def update_links(self):
        """This method updates current and previous angle of every movable link"""
        for link in self.links[1:]:
            link["previous_angle"] = link["angle"]
            link["angle"] = link["link"].angle - pi/2  # subtracting pi/2 in case of horizontal manipulator creator

    def simple_throw(self, force, link):
        """This method simulates the simplest throw imaginable"""
        length = link["length"]
        link["link"].apply_force_at_local_point((0, -force),
                                                (-length/2, 0))
        link["link"].apply_force_at_local_point((0, force),
                                                (length/2, 0))

    def pid_brake(self, link):
        # WORK IN PROGRESS
        if link["is set"]:
            link["link"].angular_velocity = 0  # freeze links in place

    def one_link_throw(self, force):
        if self.throw_on:
            x = self.links[-1]["position"][0]
            y = self.links[-1]["position"][1]
            length = self.links[-1]["length"]
            self.links[-1]["link"].apply_force_at_local_point((0, -force),
                                                              (x - length / 2, y))
            self.links[-1]["link"].apply_force_at_local_point((0, force),
                                                              (x + length / 2, y))
        else:
            for link in self.links[1:]:
                link["link"].velocity = 0, 0  # freeze links in place

    def horizontal_gripper_creator(self):
        # width = 6
        length = 20
        x = self.links[-1]["link"].position[0]  # x and y of the last link
        y = self.links[-1]["link"].position[1]
        a = 5
        b = 22
        width = self.dimensions[0]  # It stores width if horizontal manipulator creator
        gripper_body = pymunk.Body(mass=2, moment=2)
        gripper_body.position = (x - self.links[-1]["length"]/2 + a + b/2,
                                 y - width/2 - a/2)
        # vs1 = [(-a - b/2, -a/2), (a + b/2, -a/2), (a + b/2, a/2 + b), (b/2, a/2 + b), (b/2, a/2), (-b/2, a/2),
        #       (-b/2, a/2 + b), (-a - b/2, a/2 + b)]
        vs1 = [(-b/2, -a/2), (b/2, -a/2), (b/2, a/2), (-b/2, a/2)]
        box1 = pymunk.Poly(body=gripper_body, vertices=vs1, radius=1)
        vs2 = [(b/2, -a/2), (b/2 + a, -a/2), (b/2 + a, a/2 + b), (b/2, a/2 + b)]
        box2 = pymunk.Poly(body=gripper_body, vertices=vs2, radius=1)
        vs3 = [(-a - b/2, -a/2), (-b/2, -a/2), (-b/2, a/2 + b), (-a - b/2, a/2 + b)]
        box3 = pymunk.Poly(body=gripper_body, vertices=vs3, radius=1)

        left_pin_joint = pymunk.PinJoint(self.links[-1]["link"],
                                         gripper_body,
                                         (-self.links[-1]["length"]/2 + a/2, 0),
                                         (-a/2 - b/2, 0))
        left_pin_joint.error_bias = 0

        right_pin_joint = pymunk.PinJoint(self.links[-1]["link"],
                                          gripper_body,
                                          (-self.links[-1]["length"]/2 + a + b, 0),
                                          (b/2 + a/2, 0))
        right_pin_joint.error_bias = 0

        left_groove_joint = pymunk.GrooveJoint(self.links[-1]["link"],
                                               gripper_body,
                                               (-self.links[-1]["length"]/2 + a/2, width/2 - 1),
                                               (-self.links[-1]["length"]/2 + a/2, -width/2 + 1),
                                               (-b/2 - a/2, 0))
        left_groove_joint.error_bias = 0

        right_groove_joint = pymunk.GrooveJoint(self.links[-1]["link"],
                                                gripper_body,
                                                (-self.links[-1]["length"]/2 + 1.5*a + b, width/2 - 1),
                                                (-self.links[-1]["length"]/2 + 1.5*a + b, -width/2 + 1),
                                                (b/2 + a/2, 0))
        right_groove_joint.error_bias = 0

        self.space.add(gripper_body, box1, box2, box3,
                       left_pin_joint, right_pin_joint)
        # self.space.add(left_groove_joint, right_groove_joint)

        # left_claw = pymunk.Body(0.1, pymunk.moment_for_box(0.1, (width, length)))
        # left_claw.position = (self.links[-1]["link"].position[0] - self.links[-1]["length"]/2 + width/2 - 10,  # + 3
        #                       self.links[-1]["link"].position[1] - self.dimensions[0])
        #
        # left_claw_shape = pymunk.Poly.create_box(body=left_claw, size=(width, length), radius=0)
        # left_claw_shape.collision_type = self.link_collision_type
        # left_claw_shape.friction = 1.0
        #
        # left_claw_joint = pymunk.PinJoint(left_claw,
        #                                   self.links[-1]["link"],
        #                                   (0, length/2 - 1),
        #                                   (-self.links[-1]["length"]/2 + width/2 - 2, 0))  # - 2
        # left_claw_joint.error_bias = 0
        # # left_claw_joint.collide_bodies = False
        # ################### self.space.add(left_claw, left_claw_shape, left_claw_joint)
        #
        # right_claw = pymunk.Body(0.1, pymunk.moment_for_box(0.1, (width, length)))
        # right_claw.position = (self.links[-1]["link"].position[0] - self.links[-1]["length"] / 2 + width / 2 +
        #                        2 * self.ball_radius + width - 2,  # - 2
        #                        self.links[-1]["link"].position[1] - self.dimensions[0])
        #
        # right_claw_shape = pymunk.Poly.create_box(body=right_claw, size=(width, length), radius=0)
        # right_claw_shape.collision_type = self.link_collision_type
        # right_claw_shape.friction = 1.0
        #
        # right_claw_joint = pymunk.PinJoint(right_claw,
        #                                    self.links[-1]["link"],
        #                                    (0, length / 2 - 1),
        #                                    (-self.links[-1]["length"] / 2 + width / 2 +
        #                                     2 * self.ball_radius + width + 2, 0))  # + 2
        # right_claw_joint.error_bias = 0
        # right_claw_joint.collide_bodies = False

        ################### self.space.add(right_claw, right_claw_shape, right_claw_joint)

        # right_claw_groove_joint = pymunk.GrooveJoint(self.links[-1]["link"],
        #                                              right_claw,
        #                                              (-self.links[-1]["length"] / 2 + width / 2 +
        #                                               2 * self.ball_radius + width, 0),
        #                                              (-self.links[-1]["length"] / 2 + width / 2 +
        #                                               2 * self.ball_radius + width + 2, -20,),  # + 2
        #                                              (0, 0))
        # right_claw_groove_joint.error_bias = 0

        ################### self.space.add(right_claw_groove_joint)

        # left_claw_groove_joint = pymunk.GrooveJoint(self.links[-1]["link"],
        #                                             left_claw,
        #                                             (-self.links[-1]["length"]/2 + width/2, 0),
        #                                             (-self.links[-1]["length"]/2 + width/2 - 20 - 2, -20),  # - 2
        #                                             (0, 0))
        # left_claw_groove_joint.error_bias = 0

        ################### self.space.add(left_claw_groove_joint)

        # self.horizontal_ball_creator(width=width)

    def horizontal_ball_creator(self, width):
        """This method creates the ball that is to be thrown"""
        (x, y) = self.links[-1]["link"].position
        self.ball.position = (x - self.links[-1]["length"] / 2 + width / 2 +
                              self.ball_radius + width / 2, y - self.dimensions[0] / 2 - self.ball_radius)
        self.space.add(self.ball, self.ball_shape)

    def get_angles(self):
        return self.links[-1]["link"].angle

    def ball_hit_ground(self, arbiter, space, data):
        self.ball_hit_the_ground = True
        return True

    def link_reversed(self, arbiter, space, data):
        self.link_is_reversed = True
        return True

    def obstacle_hit(self, arbiter, space, data):
        self.obstacle_is_hit = True
        return True
