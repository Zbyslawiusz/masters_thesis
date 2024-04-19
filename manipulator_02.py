import pymunk
from pymunk.vec2d import Vec2d
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
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.001  # 0.03
        # self.kp = 10
        # self.ki = 0.5
        # self.kd = 0.09

        self.prev_error = 0
        self.integral = 0
        self.claw_motor_rate = -5  # Makes robotic gripper claws hold the ball

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
            "desired angle": 0,
            })
        # self.horizontal_manipulator_creator()

    def pid_force_calculator(self, error, dt):
        """This method calculates momentum that is used to rotate links based on error between current angle and
        desired angle from interpolated GA angles and timestamps"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        force = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        # print(force)
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
                "desired angle": 0,
            })
            self.space.add(link, link_shape, link_joint)

            y -= length

    def horizontal_manipulator_creator_stiff_grabber(self):
        """This method creates links of the manipulator. Links are laying on their left side upon creation"""
        mass = self.first_link_mass
        width = self.dimensions[1]  #
        length = self.dimensions[0]
        y = self.y - width/2  # x and y represent coordinates of pivot joints, not center of link coordinates
        x = self.x

        last_link = False

        for i in range(1, self.num_of_links):
            if i == self.num_of_links - 1:
                last_link = True
            # Values of length and width are inverted in case of horizontal manipulator creator
            mass *= self.mass_and_length_red
            width *= self.mass_and_length_red
            # length *= self.mass_and_length_red
            # y -= length/2

            link = pymunk.Body(mass, pymunk.moment_for_box(mass, (width, length)))
            com_xcor = x - width/2  # Center of mass x and y coordinates
            com_ycor = y
            link.position = com_xcor, com_ycor

            if not last_link:
                link_shape = pymunk.Poly.create_box(body=link, size=(width, length), radius=0)
                link_shape.collision_type = self.link_collision_type
                link_shape.friction = 1.0
                # link_shape.filter = pymunk.ShapeFilter(group=1)
            else:
                w = 4  # Width of each claw
                s = self.ball_radius * 2  # Space for the ball to fit in between two claws of the gripper
                h = self.ball_radius * 1.4  # Height of the claws

                vs1 = [(-width / 2, length / 2), (width / 2, length / 2),
                       (width / 2, -length / 2), (-width / 2, -length / 2)]
                link_box = pymunk.Poly(body=link, vertices=vs1, radius=1)
                vs2 = [(-width / 2, -length / 2), (-width / 2 + w, -length / 2),
                       (-width / 2 + w, -length / 2 - h), (-width / 2, -length / 2 - h)]
                left_claw = pymunk.Poly(body=link, vertices=vs2, radius=1)
                vs3 = [(-width / 2 + w + s, -length / 2), (-width / 2 + (2 * w) + s, -length / 2),
                       (-width / 2 + (2 * w) + s, -length / 2 - h), (-width / 2 + w + s, -length / 2 - h)]
                right_claw = pymunk.Poly(body=link, vertices=vs3, radius=1)

                ball_xcor = x - width + w + s / 2
                ball_ycor = y - length / 2 - self.ball_radius

            link_joint = pymunk.PivotJoint(self.links[i - 1]["link"],  # a
                                           link,  # b
                                           (x, y))  # pivot point
            link_joint.error_bias = 0
            link_joint.collide_bodies = False

            self.links.append({
                "link": link,
                "angle": 0,
                "previous_angle": 0,
                "length": width,
                "force": 0,
                "is set": False,
                "desired angle": 0,
            })
            if not last_link:
                self.space.add(self.links[-1]["link"], link_shape, link_joint)
            else:
                self.space.add(self.links[-1]["link"], link_box, left_claw, right_claw, link_joint)
                self.ball_creator(x=ball_xcor, y=ball_ycor)

            x -= width

    def horizontal_manipulator_creator_robotic_grabber(self):
        """This method creates links of the manipulator. Links are laying on their left side upon creation"""
        mass = self.first_link_mass
        width = self.dimensions[1]  #
        length = self.dimensions[0]
        y = self.y - width/2  # x and y represent coordinates of pivot joints, not center of link coordinates
        x = self.x

        for i in range(1, self.num_of_links):
            # Values of length and width are inverted in case of horizontal manipulator creator
            mass *= self.mass_and_length_red
            width *= self.mass_and_length_red
            # length *= self.mass_and_length_red
            # y -= length/2

            link = pymunk.Body(mass, pymunk.moment_for_box(mass, (width, length)))
            com_xcor = x - width/2  # Center of mass x and y coordinates
            com_ycor = y
            link.position = com_xcor, com_ycor

            link_shape = pymunk.Poly.create_box(body=link, size=(width, length), radius=0)
            link_shape.collision_type = self.link_collision_type
            link_shape.friction = 1.0
            # link_shape.filter = pymunk.ShapeFilter(group=1)

            link_joint = pymunk.PivotJoint(self.links[i - 1]["link"],  # a
                                           link,  # b
                                           (x, y))  # pivot point
            link_joint.error_bias = 0
            link_joint.collide_bodies = False

            self.links.append({
                "link": link,
                "angle": 0,
                "previous_angle": 0,
                "length": width,
                "force": 0,
                "is set": False,
                "desired angle": 0,
            })
            self.space.add(self.links[-1]["link"], link_shape, link_joint)

            x -= width

        w = 4  # Width of each claw
        s = self.ball_radius * 2 + 6  # Space for the ball to fit in between two claws of the gripper
        h = self.ball_radius * 2.5  # Height of the claws

        left_claw = pymunk.Body(0.2, pymunk.moment_for_box(0.2, (w, h)))
        left_claw.position = com_xcor - width/2 + w/2, com_ycor - length/2 - h/2

        left_claw_shape = pymunk.Poly.create_box(body=left_claw, size=(w, h), radius=0)
        left_claw_shape.collision_type = self.link_collision_type
        left_claw_shape.friction = 1.0

        p1 = Vec2d(com_xcor - width/2 + w/2, com_ycor - length/2)
        left_claw_joint = pymunk.PivotJoint(self.links[-1]["link"],  # a
                                            left_claw,  # b
                                            p1)  # pivot point
        left_claw_joint.error_bias = 0
        self.left_claw_motor = pymunk.SimpleMotor(a=self.links[-1]["link"],
                                                  b=left_claw,
                                                  rate=self.claw_motor_rate)
        self.left_claw_motor.max_force = 1_000_000

        right_claw = pymunk.Body(0.2, pymunk.moment_for_box(0.2, (w, h)))
        right_claw.position = com_xcor - width/2 + 1.5*w + s, com_ycor - length/2 - h / 2

        right_claw_shape = pymunk.Poly.create_box(body=right_claw, size=(w, h), radius=0)
        right_claw_shape.collision_type = self.link_collision_type
        right_claw_shape.friction = 1.0

        p2 = Vec2d(com_xcor - width/2 + 1.5*w + s, com_ycor - length/2)
        right_claw_joint = pymunk.PivotJoint(self.links[-1]["link"],  # a
                                             right_claw,  # b
                                             p2)  # pivot point
        right_claw_joint.error_bias = 0
        self.right_claw_motor = pymunk.SimpleMotor(a=self.links[-1]["link"],
                                                   b=right_claw,
                                                   rate=-self.claw_motor_rate)
        self.right_claw_motor.max_force = 1_000_000

        self.space.add(left_claw, left_claw_shape, left_claw_joint, self.left_claw_motor)
        self.space.add(right_claw, right_claw_shape, right_claw_joint, self.right_claw_motor)

        ball_xcor = com_xcor - width/2 + w + s/2
        ball_ycor = com_ycor - length/2 - self.ball_radius

        self.ball_creator(x=ball_xcor, y=ball_ycor)


    def update_links(self):
        """This method updates current and previous angle of every movable link"""
        for link in self.links[1:]:
            angle = link["link"].angle
            if True:  # Placeholder, means horizontal manipulator is used
                angle -= pi/2  # subtracting pi/2 in case of horizontal manipulator creator
            link["previous_angle"] = link["angle"]
            if angle > 2*pi:
                angle = 0 + angle % (2*pi)
            elif angle < -2*pi:
                angle = 0 - angle % (2 * pi)
            else:
                pass
            link["angle"] = angle

    def simple_throw(self, force, link):
        """This method applies force as momentum to the passed link"""
        length = link["length"]
        link["link"].apply_force_at_local_point((0, -force),
                                                (-length/2, 0))
        link["link"].apply_force_at_local_point((0, force),
                                                (length/2, 0))

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

    def ball_creator(self, x, y):
        """This method creates the ball that is to be thrown"""
        self.ball.position = x, y
        self.space.add(self.ball, self.ball_shape)

    def ball_hit_ground(self, arbiter, space, data):
        self.ball_hit_the_ground = True
        return True

    def obstacle_hit(self, arbiter, space, data):
        self.obstacle_is_hit = True
        return True

    def ball_not_touching_gripper(self, arbiter, space, data):
        return True
