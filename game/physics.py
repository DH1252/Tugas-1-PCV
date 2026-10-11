import numpy as np
import pymunk

ITEM, HAND, WALL = 1, 2, 3
TOUCH_ONLY = ("spike", "star")


class World:
    def __init__(self, p, size):
        self.p = p
        self.size = size
        self.space = pymunk.Space()
        self.space.gravity = (0, p["gravity"])
        self.gravity_mult = 1.0
        self.items = []
        self.hands = {}
        self.events = []
        w, h = size
        top = -70
        for a, b in (((0, top), (0, h + 400)), ((w, top), (w, h + 400)), ((-50, top), (w + 50, top))):
            seg = pymunk.Segment(self.space.static_body, a, b, 1)
            seg.elasticity = p["restitution_wall"]
            seg.friction = 0.2
            seg.collision_type = WALL
            self.space.add(seg)
        self.space.on_collision(HAND, ITEM, begin=self._begin, post_solve=self._post_solve)

    def set_gravity_mult(self, mult):
        self.gravity_mult = mult
        self.space.gravity = (0, self.p["gravity"] * mult)

    def add(self, item):
        mass = np.pi * item.radius ** 2 / 100 * (2.0 if item.kind == "heavy" else 1.0)
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, item.radius))
        body.position = tuple(item.pos)
        body.velocity = tuple(item.vel)
        body.velocity_func = self._velocity_func(item)
        shape = pymunk.Circle(body, item.radius)
        shape.elasticity = item.bounce
        shape.friction = 0.3
        shape.collision_type = ITEM
        shape.item = item
        item.body, item.shape = body, shape
        self.space.add(body, shape)
        self.items.append(item)

    def remove(self, item):
        if item in self.items:
            self.space.remove(item.body, item.shape)
            self.items.remove(item)

    def clear(self):
        for item in list(self.items):
            self.remove(item)

    def _velocity_func(self, item):
        max_speed = self.p["max_speed"]

        def update(body, gravity, damping, dt):
            g = (gravity[0] * item.gravity, gravity[1] * item.gravity)
            pymunk.Body.update_velocity(body, g, damping * max(0.0, 1.0 - item.drag * dt), dt)
            speed = body.velocity.length
            if speed > max_speed:
                body.velocity = body.velocity * (max_speed / speed)
        return update

    def _begin(self, arbiter, space, data):
        item = arbiter.shapes[1].item
        if item.kind in TOUCH_ONLY:
            arbiter.process_collision = False
            if item.alive:
                item.alive = False
                self.events.append((item.kind, item))

    def _post_solve(self, arbiter, space, data):
        item = arbiter.shapes[1].item
        if not arbiter.is_first_contact or item.kind in TOUCH_ONLY:
            return
        if arbiter.normal.y < 0:
            vx, vy = item.body.velocity
            item.body.velocity = (vx, min(vy, -self.p["min_up_speed"] * item.bounce))
        if item.cooldown <= 0:
            self.events.append(("hit", item))
            item.cooldown = self.p["hit_cooldown"]

    def _sync_hands(self, paddles):
        seen = set()
        for pd in paddles:
            seen.add(pd.id)
            if pd.id not in self.hands:
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = tuple(pd.prev)
                shape = pymunk.Circle(body, pd.radius)
                shape.elasticity = self.p["restitution_hand"]
                shape.friction = 0.6
                shape.collision_type = HAND
                self.space.add(body, shape)
                self.hands[pd.id] = (body, shape)
            body, shape = self.hands[pd.id]
            if abs(shape.radius - pd.radius) > 0.5:
                shape.unsafe_set_radius(pd.radius)
        for hid in [h for h in self.hands if h not in seen]:
            self.space.remove(*self.hands.pop(hid))

    def step(self, dt, paddles):
        self.events = []
        self._sync_hands(paddles)
        n = max(1, int(self.p["substeps"]))
        h = dt / n
        limit = self.p["max_hand_speed"]
        for i in range(n):
            f = (i + 1) / n
            for pd in paddles:
                body = self.hands[pd.id][0]
                target = pd.prev + (pd.pos - pd.prev) * f
                vel = (target - np.array(body.position)) / h
                speed = float(np.hypot(*vel))
                if speed > limit:
                    vel = vel * (limit / speed)
                body.velocity = tuple(vel)
            self.space.step(h)
            for item in self.items:
                item.cooldown = max(0.0, item.cooldown - h)
        height = self.size[1]
        for item in list(self.items):
            item.pos[:] = item.body.position
            item.vel[:] = item.body.velocity
            if item.alive and item.pos[1] - item.radius > height:
                item.alive = False
                self.events.append(("drop", item))
            if not item.alive:
                self.remove(item)
        return self.events
