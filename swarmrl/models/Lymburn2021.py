import swarmrl as srl
from swarmrl.models.interaction_model import Action
import numpy as np


class MyForceModel(srl.models.InteractionModel):
    def __init__(self,
                 force_params: dict,
                 detection_radius_position_colls=np.inf,
                 detection_radius_position_pred=np.inf,
                 home_pos=np.array([500, 500, 0]),
                 agent_speed=10
                 ):
        self.force_params = force_params

        self.detection_radius_position_colls = detection_radius_position_colls
        # r_alignment=r_repulsion,
        # split in 2 if r_a not= r_r, split colls_in_vision as well
        self.detection_radius_position_pred = detection_radius_position_pred
        self.home_pos = home_pos
        self.agent_speed = agent_speed
        self.t = 0

    def calc_action(self, colloids):
        actions = []
        self.t += 0.2 / 5
        for colloid in colloids:
            if colloid.type == 1:
                y, x = colloid.pos[1], colloid.pos[0]
                force_x = 100 * np.cos(0.1 * self.t)
                force_y = self.home_pos[1] - y
                nd = np.array([force_x, force_y, 0])
                new_direction = nd / np.linalg.norm(nd)
                actions.append(Action(force=50*np.linalg.norm(nd), new_direction=new_direction))
                continue

            other_colloids = [c for c in colloids if c is not colloid and not c.type == 1]
            colls_in_vision = get_colloids_in_vision(
                colloid,
                other_colloids,
                vision_radius=self.detection_radius_position_colls)

            predator = [p for p in colloids if p is p.type == 1]  # only one predator is taken in account
            pred_in_vision = get_colloids_in_vision(
                colloid,
                predator,
                vision_radius=self.detection_radius_position_pred)

            colls_in_vision_position = np.array([c.pos for c in colls_in_vision])
            # colls_in_vision_director = np.array([c.director for c in colls_in_vision])
            colls_in_vision_velocity = np.array([c.velocity for c in colls_in_vision])
            pred_in_vision_position = np.array([p.pos for p in pred_in_vision])

            force_a, force_r = np.array([0, 0, 0]), np.array([0, 0, 0])
            if len(colls_in_vision) > 0:
                force_a = np.sum(colls_in_vision_velocity - colloid.velocity, axis=0)

                force_r_notnorm = np.sum(colls_in_vision_position - colloid.pos, axis=0)
                dist_norm = np.linalg.norm(colls_in_vision_position - colloid.pos)
                force_r = force_r_notnorm / dist_norm

            force_h = self.home_pos - colloid.pos

            force_p = np.array([0, 0, 0])
            if len(pred_in_vision) > 0:
                force_p_notnorm = np.sum(colloid.pos - pred_in_vision_position, axis=0)
                dist_norm_pred = np.linalg.norm(colloid.pos - pred_in_vision_position)
                force_p = force_p_notnorm / dist_norm_pred

            force_f = -colloid.velocity * (np.abs(colloid.velocity) - self.agent_speed) / self.agent_speed

            force = self.force_params["K_a"] * force_a + self.force_params["K_r"] * force_r \
                + self.force_params["K_h"] * force_h \
                + self.force_params["K_p"] * force_p + self.force_params["K_f"] * force_f

            force_magnitude = np.linalg.norm(force)
            force_direction = force / force_magnitude

            actions.append(Action(force=force_magnitude, new_direction=force_direction))
        return actions


def get_colloids_in_vision(coll, other_coll, vision_radius):
    my_pos = coll.pos
    colls_in_vision = []
    for other_p in other_coll:
        dist = other_p.pos - my_pos
        dist_norm = np.linalg.norm(dist)
        in_range = dist_norm < vision_radius
        if not in_range:
            continue
        if in_range:
            colls_in_vision.append(other_p)
    return colls_in_vision


def rotate_vector_clockwise(v, alpha):
    r = np.array([[np.cos(alpha), np.sin(alpha)],
                  [-np.sin(alpha), np.cos(alpha)]])
    return np.matmul(r, v)