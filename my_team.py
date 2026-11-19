# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, Queue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

# Our team has one attacker and one defender. Both of them work the same way:
# every turn they pick a target (some food, home, an invader...) and then use
# a BFS on the maze to find the first step towards it, treating dangerous
# squares as if they were walls. It's the same BFS idea as in Lab 1, we just
# run it every turn because the ghosts keep moving.

# A ghost whose scared timer is above this is harmless for a few more moves,
# so the attacker doesn't need to run away from it.
SAFE_SCARED_TIME = 3

# How far away a ghost has to be before the attacker stops worrying about it.
DANGER_DISTANCE = 5


class SmartAgent(CaptureAgent):
    """
    Common code for both agents: where home is, the BFS, and a few helpers
    to know where the enemies are.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.boundary = []

    def register_initial_state(self, game_state):
        # The parent computes all the maze distances here (we have 15 seconds
        # at the start), so later get_maze_distance is just a lookup.
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()

        # The "boundary" is the column on our side right next to the middle.
        # Reaching any of these squares means the food we carry is safe.
        width, height = self.walls.width, self.walls.height
        home_x = width // 2 - 1 if self.red else width // 2
        self.boundary = [(home_x, y) for y in range(height) if not self.walls[home_x][y]]

    # ---------- small helpers ----------

    def my_position(self, game_state):
        return nearest_point(game_state.get_agent_position(self.index))

    def on_our_side(self, pos):
        # The red team owns the left half of the map and blue the right half.
        half = self.walls.width // 2
        return pos[0] < half if self.red else pos[0] >= half

    def distance_home(self, pos):
        return min(self.get_maze_distance(pos, b) for b in self.boundary)

    def visible_enemies(self, game_state):
        """List of (index, position, agent_state) for the enemies we can see."""
        enemies = []
        for i in self.get_opponents(game_state):
            pos = game_state.get_agent_position(i)
            if pos is not None:
                enemies.append((i, nearest_point(pos), game_state.get_agent_state(i)))
        return enemies

    def moves_left(self, game_state):
        # timeleft counts the moves of all 4 agents, so we get a quarter of it.
        return game_state.data.timeleft // 4

    # ---------- path finding ----------

    def bfs_first_step(self, game_state, targets, blocked=frozenset()):
        """
        Breadth-first search from our position to the closest square in
        targets, never going through the squares in blocked. Returns the
        first action of that path, or None if no target can be reached.
        Since every move costs 1, BFS already gives the shortest path.
        """
        start = self.my_position(game_state)
        targets = set(targets)
        if not targets:
            return None
        if start in targets:
            return Directions.STOP

        # For each square we remember the first action we took from the start
        # to get there, so when we reach a target we already know what to do.
        first_action = {start: None}
        queue = Queue()
        queue.push(start)
        while not queue.is_empty():
            pos = queue.pop()
            for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
                dx, dy = Actions.direction_to_vector(action)
                nxt = (int(pos[0] + dx), int(pos[1] + dy))
                if self.walls[nxt[0]][nxt[1]] or nxt in first_action or nxt in blocked:
                    continue
                first_action[nxt] = first_action[pos] or action
                if nxt in targets:
                    return first_action[nxt]
                queue.push(nxt)
        return None

    def neighbours(self, pos):
        """The square itself plus the squares one step away (not walls)."""
        x, y = pos
        cells = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in cells if not self.walls[c[0]][c[1]]]

    def safe_fallback(self, game_state, threats):
        """
        When BFS can't find a path, just take the legal move that keeps us as
        far as possible from the threats (or a random one if there are none).
        """
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        if not actions:
            return Directions.STOP
        if not threats:
            return random.choice(actions)

        def score(action):
            dx, dy = Actions.direction_to_vector(action)
            pos = self.my_position(game_state)
            nxt = (int(pos[0] + dx), int(pos[1] + dy))
            return min(self.get_maze_distance(nxt, t) for t in threats)

        return max(actions, key=score)


class OffensiveAgent(SmartAgent):
    """
    The attacker goes to the enemy side, eats food and brings it back.
    It decides every turn between "keep eating" and "go home", and it never
    walks next to a ghost that can eat it.
    """

    def choose_action(self, game_state):
        my_pos = self.my_position(game_state)
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        # Ghosts that can actually eat us: enemies on their own side that are
        # not scared (or will stop being scared very soon).
        enemies = self.visible_enemies(game_state)
        ghosts = [pos for _, pos, st in enemies
                  if not st.is_pacman and st.scared_timer <= SAFE_SCARED_TIME]
        close_ghosts = [g for g in ghosts if self.get_maze_distance(my_pos, g) <= DANGER_DISTANCE]

        # Squares where a ghost could catch us next turn. We only care about
        # the enemy side: on our side we are a ghost and they can't eat us.
        blocked = set()
        for g in ghosts:
            blocked.update(c for c in self.neighbours(g) if not self.on_our_side(c))

        # If we are at home (still a ghost) and an invader is right next to
        # us, we might as well eat it on the way.
        if not my_state.is_pacman and my_state.scared_timer == 0:
            invaders = [pos for _, pos, st in enemies if st.is_pacman
                        and self.get_maze_distance(my_pos, pos) <= 2]
            if invaders:
                action = self.bfs_first_step(game_state, invaders)
                if action is not None:
                    return action

        dist_home = self.distance_home(my_pos)

        # When should we go back? If we carry food and a ghost is near, if we
        # carry a lot already, if the game is about to end, or if there's
        # almost no food left (the game ends when only 2 dots remain).
        go_home = carrying > 0 and (
            close_ghosts
            or carrying >= self.carry_limit(food)
            or self.moves_left(game_state) <= dist_home + 5
            or len(food) <= 2)

        if go_home:
            # Running home with a ghost behind us: a capsule on the way is
            # even better, since it makes the ghosts scared.
            targets = list(self.boundary)
            if close_ghosts:
                targets += [c for c in capsules
                            if self.get_maze_distance(my_pos, c) < dist_home]
            action = self.bfs_first_step(game_state, targets, blocked)
            if action is not None:
                return action
            return self.safe_fallback(game_state, ghosts)

        # Otherwise keep eating. With a ghost close by we only go for food we
        # can reach before the ghost does, so we don't walk into a trap.
        targets = food
        if close_ghosts:
            targets = [f for f in food
                       if all(self.get_maze_distance(my_pos, f) < self.get_maze_distance(g, f)
                              for g in close_ghosts)]
            targets += capsules

        action = self.bfs_first_step(game_state, targets, blocked)
        if action is not None:
            return action

        # Nothing safe to eat: retreat home and try again later.
        action = self.bfs_first_step(game_state, self.boundary, blocked)
        if action is not None:
            return action
        return self.safe_fallback(game_state, ghosts)

    def carry_limit(self, food):
        # How many dots we carry before going back. Carrying more is
        # risky because dying drops all of it, so we keep the number small,
        # and even smaller when there's not much food left anyway.
        return max(2, min(6, len(food) // 4))


class DefensiveAgent(SmartAgent):
    """
    The defender stays on our side. It chases any invader it can see; if it
    can't see one but food goes missing, it goes to where the food was eaten;
    otherwise it waits near the middle, close to our food.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_known_invader = None

    def choose_action(self, game_state):
        my_pos = self.my_position(game_state)
        my_state = game_state.get_agent_state(self.index)

        # The defender never crosses to the enemy side, so the whole enemy
        # half counts as blocked for its BFS.
        blocked = {(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
                   if not self.walls[x][y] and not self.on_our_side((x, y))}

        invaders = [pos for _, pos, st in self.visible_enemies(game_state) if st.is_pacman]

        if invaders:
            self.last_known_invader = min(invaders, key=lambda p: self.get_maze_distance(my_pos, p))
            if my_state.scared_timer > 0:
                # While we're scared the invader can eat us, so we follow it
                # from a distance instead of touching it.
                for inv in invaders:
                    blocked.update(self.neighbours(inv))
                targets = [c for inv in invaders for c in self.ring_around(inv, 2)]
            else:
                targets = invaders
            action = self.bfs_first_step(game_state, targets, blocked)
            if action is not None:
                return action

        # We can't see anyone, but if one of our dots disappeared since last
        # turn, an invader just ate it, so we head there.
        eaten = self.recently_eaten_food(game_state)
        if eaten:
            self.last_known_invader = min(eaten, key=lambda p: self.get_maze_distance(my_pos, p))

        if self.last_known_invader is not None:
            if my_pos == self.last_known_invader:
                # We got there and there's nothing: forget about it.
                self.last_known_invader = None
            else:
                action = self.bfs_first_step(game_state, [self.last_known_invader], blocked)
                if action is not None:
                    return action

        # Nothing going on: wait at the boundary square that is closest to
        # our food, so we're ready when someone comes in.
        action = self.bfs_first_step(game_state, [self.patrol_point(game_state)], blocked)
        if action is not None:
            return action
        return self.safe_fallback(game_state, invaders)

    def recently_eaten_food(self, game_state):
        previous = self.get_previous_observation()
        if previous is None:
            return []
        before = set(self.get_food_you_are_defending(previous).as_list())
        now = set(self.get_food_you_are_defending(game_state).as_list())
        return list(before - now)

    def patrol_point(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.boundary[len(self.boundary) // 2]
        # The boundary square with the smallest total distance to our food is
        # a good place to stand: it's roughly in front of most of it.
        return min(self.boundary,
                   key=lambda b: sum(self.get_maze_distance(b, f) for f in food))

    def ring_around(self, pos, radius):
        """Squares on our side at exactly this maze distance from pos."""
        cells = []
        for x in range(pos[0] - radius, pos[0] + radius + 1):
            for y in range(pos[1] - radius, pos[1] + radius + 1):
                if (0 <= x < self.walls.width and 0 <= y < self.walls.height
                        and not self.walls[x][y] and self.on_our_side((x, y))
                        and self.get_maze_distance(pos, (x, y)) == radius):
                    cells.append((x, y))
        return cells
