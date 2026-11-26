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
from contest.util import nearest_point, manhattan_distance, Queue


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

# Our team has one attacker and one defender, but the roles can change during
# the game depending on the score (see choose_role). Both agents work the same
# way: every turn they pick a target (some food, home, an invader...) and then
# use a BFS on the maze to find the first step towards it, treating dangerous
# squares as if they were walls. It's the same BFS idea as in Lab 1, we just
# run it every turn because the ghosts keep moving.

# A ghost whose scared timer is above this is harmless for a few more moves,
# so the attacker doesn't need to run away from it.
SAFE_SCARED_TIME = 3

# How far away a ghost has to be before the attacker stops worrying about it.
DANGER_DISTANCE = 5

# Enemies closer than this (Manhattan distance) to one of our agents are
# visible. If we can't see an enemy, we know it is further away than this.
SIGHT_RANGE = 5

# When we are winning and there are fewer moves than this left, both agents
# stay home to protect the lead.
ENDGAME_MOVES = 75


class SmartAgent(CaptureAgent):
    """
    Common code for both agents: where home is, the BFS, the dead ends of the
    maze, where we think the enemies are, and the attack and defend behaviours.
    The subclasses only decide which of the two behaviours to use each turn.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # Everything that depends on the map is filled in
        # register_initial_state, when we get to see the layout.
        self.start = None               # our start square
        self.walls = None               # grid with True where there is a wall
        self.boundary = []              # our squares right next to the middle
        self.cells = []                 # every square that is not a wall
        self.enemy_half = frozenset()   # every square on the enemy side
        self.dead_ends = {}             # square -> (exit, depth), see compute_dead_ends
        self.beliefs = {}               # enemy index -> {square: probability}
        self.last_known_invader = None  # last place where we saw / suspect an invader

    def register_initial_state(self, game_state):
        # The parent computes all the maze distances here (we have 15 seconds
        # at the start), so later get_maze_distance is just a lookup.
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        width, height = self.walls.width, self.walls.height

        # All the squares that are not walls.
        self.cells = [(x, y) for x in range(width) for y in range(height) if not self.walls[x][y]]

        # The "boundary" is the column on our side right next to the middle.
        # Reaching any of these squares means the food we carry is safe.
        home_x = width // 2 - 1 if self.red else width // 2
        self.boundary = [(home_x, y) for y in range(height) if not self.walls[home_x][y]]

        # The enemy half never changes, so we compute it once here. The
        # defender treats it as blocked so it doesn't cross the middle.
        self.enemy_half = frozenset(c for c in self.cells if not self.on_our_side(c))

        # The dead ends don't change either, and finding them takes a few
        # passes over the whole map, so we also do it once at the start.
        self.dead_ends = self.compute_dead_ends()

        # At the start we know exactly where each enemy is: its start square.
        for enemy in self.get_opponents(game_state):
            self.beliefs[enemy] = {game_state.get_initial_agent_position(enemy): 1.0}

    def choose_action(self, game_state):
        # First update where we think the enemies are, then pick a role for
        # this turn and act according to it.
        self.update_beliefs(game_state)
        if self.choose_role(game_state) == 'attack':
            return self.attack(game_state)
        return self.defend(game_state)

    def choose_role(self, game_state):
        """Returns 'attack' or 'defend'. Each subclass decides differently."""
        raise NotImplementedError

    # ---------- small helpers ----------

    def my_position(self, game_state):
        # Positions can be floats while an agent is between two squares, so
        # we round to the nearest square to use it as a key in the BFS.
        return nearest_point(game_state.get_agent_position(self.index))

    def on_our_side(self, pos):
        # The red team owns the left half of the map and blue the right half.
        half = self.walls.width // 2
        return pos[0] < half if self.red else pos[0] >= half

    def distance_home(self, pos):
        # Number of moves from pos to the closest square of our boundary.
        # For an enemy position it tells how close it is to entering our side.
        return min(self.get_maze_distance(pos, b) for b in self.boundary)

    def visible_enemies(self, game_state):
        """List of (index, position, agent_state) for the enemies we can see."""
        enemies = []
        for i in self.get_opponents(game_state):
            pos = game_state.get_agent_position(i)
            if pos is not None:
                enemies.append((i, nearest_point(pos), game_state.get_agent_state(i)))
        return enemies

    def enemy_is_invading(self, game_state):
        # Even if we can't see an enemy, the game tells us whether it is a
        # Pacman (i.e. on our side) or a ghost. Only its position is hidden.
        return any(game_state.get_agent_state(i).is_pacman for i in self.get_opponents(game_state))

    def moves_left(self, game_state):
        # timeleft counts the moves of all 4 agents, so we get a quarter of it.
        return game_state.data.timeleft // 4

    def winning_at_the_end(self, game_state):
        # get_score is from our point of view: positive means we are ahead.
        return self.get_score(game_state) > 0 and self.moves_left(game_state) <= ENDGAME_MOVES

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

    # ---------- dead ends ----------

    def compute_dead_ends(self):
        """
        Finds the squares that are inside a dead end (a corridor or group of
        corridors with only one way out). Returns a dictionary
        square -> (exit, depth), where exit is the square where the dead end
        joins the rest of the maze and depth is how many moves it takes to get
        from the square to that exit.

        We repeatedly remove the squares with only one open neighbour (the
        end of a corridor). Removing one can make the next square a "corridor
        end" too, so we keep going until only the squares that are part of a
        loop remain. Everything we removed is inside a dead end. Then we do a
        BFS from each remaining square into the removed ones to get the exit
        and the depth.
        """
        cells = set(self.cells)

        def open_neighbours(pos):
            x, y = pos
            return [c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if c in cells]

        # degree = how many open neighbours each square has. A square with
        # degree 1 is the end of a corridor (only one way in and out).
        degree = {c: len(open_neighbours(c)) for c in cells}
        removed = set()

        # We start with all the corridor ends and remove them one by one.
        queue = Queue()
        for c in cells:
            if degree[c] <= 1:
                queue.push(c)
        while not queue.is_empty():
            c = queue.pop()
            if c in removed:
                continue
            removed.add(c)
            # Removing c takes one neighbour away from the squares next to it.
            # If one of them is left with a single neighbour, it has become a
            # corridor end too, so it also goes into the queue.
            for n in open_neighbours(c):
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] == 1:
                        queue.push(n)

        # If the whole maze got removed it has no loops at all, so "dead end"
        # means nothing useful there.
        if removed == cells:
            return {}

        # The squares that were not removed are part of loops (you can always
        # get out of them in two directions). A removed square next to one of
        # them is the entrance of a dead end, and that loop square is its exit.
        dead_ends = {}
        for exit_square in cells - removed:
            for first in open_neighbours(exit_square):
                if first not in removed or first in dead_ends:
                    continue
                # BFS inside this dead end, counting the distance to the exit.
                dead_ends[first] = (exit_square, 1)
                queue = Queue()
                queue.push(first)
                while not queue.is_empty():
                    pos = queue.pop()
                    for n in open_neighbours(pos):
                        if n in removed and n not in dead_ends:
                            dead_ends[n] = (exit_square, dead_ends[pos][1] + 1)
                            queue.push(n)
        return dead_ends

    def is_trap(self, my_pos, target, ghosts):
        """
        True if going to target means entering a dead end that a ghost could
        close before we get back out. To be safe we need to go to the target
        and come back to the exit before any ghost reaches that exit.
        """
        # Food that is not in a dead end can't trap us (there is always
        # another way out), and with no ghost around there is no danger.
        if target not in self.dead_ends or not ghosts:
            return False
        exit_square, depth = self.dead_ends[target]

        # Moves we need: go to the target, then walk back out to the exit.
        time_needed = self.get_maze_distance(my_pos, target) + depth

        # If a ghost can get to the exit in that time (+1 to be safe), it can
        # wait there and we would be stuck inside.
        return any(self.get_maze_distance(g, exit_square) <= time_needed + 1 for g in ghosts)

    # ---------- where are the enemies? ----------

    def update_beliefs(self, game_state):
        """
        Keeps, for each enemy, a probability for every square of being where
        that enemy is (a simple filter, like the inference seen in class).
        Every turn:
          1. If we see the enemy, we know its square exactly.
          2. Otherwise the enemy has moved one step (or stayed): we spread
             each probability evenly to the square and its neighbours.
          3. Then we use what we observe to discard or reweight squares:
             - get_agent_distances() gives a noisy distance (true distance
               +-6), so squares that don't match it get probability 0.
             - If it's a Pacman it's on our side, otherwise on its own side.
             - If we can't see it, it's not within SIGHT_RANGE of us or our
               teammate.
        If no square is left (for example the enemy was eaten and went back
        to its start), we start again from every square that fits.
        """
        my_pos = self.my_position(game_state)
        # noisy[i] is the noisy distance from us to agent i (measured from
        # our own position, with the Manhattan distance).
        noisy = game_state.get_agent_distances()
        # We always know where we and our teammate are.
        team_positions = [game_state.get_agent_position(t) for t in self.get_team(game_state)]

        for enemy in self.get_opponents(game_state):
            # Step 1: if we can see it, there is no doubt about where it is.
            pos = game_state.get_agent_position(enemy)
            if pos is not None:
                self.beliefs[enemy] = {nearest_point(pos): 1.0}
                continue

            is_pacman = game_state.get_agent_state(enemy).is_pacman

            def fits(square):
                # Could the enemy be on this square, given what we know for
                # sure? A Pacman is always on our side and a ghost on its own
                # side, and since we can't see it, it can't be close to us.
                if self.on_our_side(square) != is_pacman:
                    return False
                if any(manhattan_distance(square, t) <= SIGHT_RANGE for t in team_positions):
                    return False
                return True

            def likelihood(square):
                # How likely it is to get this noisy reading if the enemy were
                # on this square. The game gives the formula: 1/13 if the
                # reading is within +-6 of the true distance, 0 otherwise.
                if noisy is None:
                    return 1.0
                return game_state.get_distance_prob(manhattan_distance(my_pos, square), noisy[enemy])

            # Step 2: the enemy moves. We don't know where to, so each square
            # gives its probability in equal parts to itself (staying still)
            # and to each neighbour it could move to.
            moved = {}
            for square, prob in self.beliefs[enemy].items():
                options = self.neighbours(square)
                for n in options:
                    moved[n] = moved.get(n, 0.0) + prob / len(options)

            # Step 3: we observe. Squares that don't fit are dropped, and the
            # rest are multiplied by how well they match the noisy distance.
            new_belief = {s: p * likelihood(s) for s, p in moved.items() if fits(s)}

            # If nothing is left, our belief was wrong (the enemy was eaten
            # and respawned, for example), so we start again from every
            # square that fits the observation.
            if sum(new_belief.values()) == 0:
                new_belief = {s: likelihood(s) for s in self.cells if fits(s)}

            total = sum(new_belief.values())
            if total == 0:
                # Nothing fits at all (shouldn't happen): keep the old belief.
                continue
            # Normalize so the probabilities add up to 1 again. We also drop
            # the squares with probability 0 so the dictionary stays small.
            self.beliefs[enemy] = {s: p / total for s, p in new_belief.items() if p > 0}

    def most_likely_position(self, enemy):
        # Our best guess of where the enemy is: the square with the highest
        # probability in its belief.
        belief = self.beliefs[enemy]
        return max(belief, key=belief.get)

    # ---------- attacking ----------

    def attack(self, game_state):
        """
        Go to the enemy side, eat food and bring it back. Every turn we decide
        between "keep eating" and "go home", we never walk next to a ghost
        that can eat us, and we don't enter dead ends that a ghost can close.
        """
        my_pos = self.my_position(game_state)
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying              # dots we carry but haven't scored yet
        food = self.get_food(game_state).as_list()    # the enemy's dots we can eat
        capsules = self.get_capsules(game_state)      # the enemy's capsules we can eat
        dist_home = self.distance_home(my_pos)

        # How many moves the enemy ghosts will stay scared after one of us ate
        # a capsule. The timer is known even for enemies we can't see. If it
        # is long enough to come back home afterwards, we are "powered".
        scared_left = min(game_state.get_agent_state(i).scared_timer
                          for i in self.get_opponents(game_state))
        # The +2 is a small safety margin. We use the minimum of the two
        # enemies because one of them may have been eaten and respawned, which
        # resets its timer to 0: then that ghost is dangerous again.
        powered = scared_left > dist_home + 2

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
                action = self.bfs_first_step(game_state, invaders, blocked)
                if action is not None:
                    return action

        # While the ghosts are scared we eat as much as we can, with no carry
        # limit. When the scared time is about to run out, powered becomes
        # False and the normal rules below make us go home with the food.
        # (We tried also chasing the scared ghosts to eat them, but in our
        # tests it lost more games than it won: the time spent chasing was
        # time not spent eating, so we don't do it.)
        # The extra checks are the same as for going home below: if the game
        # is about to end or almost no food is left, we stop eating and go
        # back to score what we carry.
        if powered and not ghosts:
            if self.moves_left(game_state) > dist_home + 5 and len(food) > 2:
                # No ghost is dangerous, so no blocked squares are needed.
                action = self.bfs_first_step(game_state, food)
                if action is not None:
                    return action

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

        # And we skip food inside a dead end if a nearby ghost could reach the
        # exit of the dead end before we come back out of it. We only look at
        # nearby ghosts: a ghost far away is not chasing us, and counting it
        # would make us skip almost every dead end on the map.
        targets = [t for t in targets if not self.is_trap(my_pos, t, close_ghosts)]

        action = self.bfs_first_step(game_state, targets, blocked)
        if action is not None:
            return action

        # Nothing safe to eat: retreat home and try again later.
        action = self.bfs_first_step(game_state, self.boundary, blocked)
        if action is not None:
            return action
        return self.safe_fallback(game_state, ghosts)

    def carry_limit(self, food):
        # How many dots we carry before going back. Carrying more is risky
        # because dying drops all of it, but every trip home also wastes
        # moves. We tried several limits in hundreds of test games and this
        # one won the most: up to half of the food that is left (at least 3,
        # at most 15). The attacker already runs home as soon as a ghost gets
        # close, so carrying more rarely means losing it.
        return max(3, min(15, len(food) // 2))

    # ---------- defending ----------

    def defend(self, game_state):
        """
        Stay on our side. Chase any invader we can see; if we can't see one
        but food goes missing, go to where the food was eaten; if we know an
        enemy is on our side but can't see it, go to where our filter thinks
        it is; otherwise wait at the point of the boundary where the closest
        enemy is most likely to come in.
        """
        my_pos = self.my_position(game_state)
        my_state = game_state.get_agent_state(self.index)

        # If we were attacking and had to switch to defence, first come back
        # home. On the way we avoid the ghosts, like the attacker does.
        if not self.on_our_side(my_pos):
            ghosts = [pos for _, pos, st in self.visible_enemies(game_state)
                      if not st.is_pacman and st.scared_timer <= SAFE_SCARED_TIME]
            blocked = set()
            for g in ghosts:
                blocked.update(c for c in self.neighbours(g) if not self.on_our_side(c))
            action = self.bfs_first_step(game_state, self.boundary, blocked)
            if action is not None:
                return action
            return self.safe_fallback(game_state, ghosts)

        # The defender never crosses to the enemy side, so the whole enemy
        # half counts as blocked for its BFS.
        blocked = set(self.enemy_half)

        # Enemies we can see that are on our side (they are Pacmen here).
        invaders = [pos for _, pos, st in self.visible_enemies(game_state) if st.is_pacman]

        if invaders:
            # We remember the closest one, so if it goes out of sight we can
            # still go to the last place we saw it.
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

        # An enemy is on our side but we can't see it: go to the square where
        # our filter thinks it most likely is.
        # We know it's on our side because is_pacman is True even when we
        # can't see it. The extra on_our_side check is just in case the filter
        # hasn't caught up yet (for example the turn it crosses the middle).
        hidden_invaders = [self.most_likely_position(i) for i in self.get_opponents(game_state)
                           if game_state.get_agent_state(i).is_pacman]
        hidden_invaders = [p for p in hidden_invaders if self.on_our_side(p)]
        if hidden_invaders:
            action = self.bfs_first_step(game_state, hidden_invaders, blocked)
            if action is not None:
                return action

        # Nobody inside: wait where the closest enemy will probably come in.
        action = self.bfs_first_step(game_state, [self.patrol_point(game_state)], blocked)
        if action is not None:
            return action
        return self.safe_fallback(game_state, invaders)

    def recently_eaten_food(self, game_state):
        # We compare our food in the previous turn with our food now: the dots
        # that were there before and are gone now were just eaten, so an
        # invader is (or was a moment ago) on those squares.
        previous = self.get_previous_observation()
        if previous is None:
            # First turn of the game: there is no previous turn to compare.
            return []
        before = set(self.get_food_you_are_defending(previous).as_list())
        now = set(self.get_food_you_are_defending(game_state).as_list())
        return list(before - now)

    def patrol_point(self, game_state):
        """
        The boundary square where we should wait. We take the enemy that is
        most likely closest to our side (using the filter) and wait at the
        boundary square nearest to it, so we meet it when it crosses. If the
        enemies are both far away, we wait in front of our food instead.
        """
        # Where we think each enemy is right now.
        estimates = [self.most_likely_position(i) for i in self.get_opponents(game_state)]
        if estimates:
            # The enemy that is closest to entering our side.
            closest = min(estimates, key=self.distance_home)
            # Only if it is reasonably close: when an enemy is far away its
            # estimate is not very precise, and it could still go anywhere.
            if self.distance_home(closest) <= 2 * DANGER_DISTANCE:
                return min(self.boundary, key=lambda b: self.get_maze_distance(b, closest))

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


class OffensiveAgent(SmartAgent):
    """
    Attacks by default. When we are winning and the game is about to end, it
    goes back home and defends too, to protect the lead.
    """

    def choose_role(self, game_state):
        # Near the end, keeping the points we have is worth more than trying
        # to get a few more: if an invader scores, we could lose the game.
        # If we carry food when this happens, defend() first takes us home,
        # so we still score it.
        if self.winning_at_the_end(game_state):
            return 'defend'
        return 'attack'


class DefensiveAgent(SmartAgent):
    """
    Defends by default. When we are losing and no enemy is on our side or
    close to it, there is nothing to defend, so it attacks too. As soon as an
    enemy gets close to our side it comes back to defend.
    """

    def choose_role(self, game_state):
        # We check this every turn, so the defender goes back to defending
        # as soon as one of the conditions stops being true (for example
        # when an enemy gets close to the middle).
        losing = self.get_score(game_state) < 0
        if losing and not self.enemy_is_invading(game_state) and self.enemies_far_away(game_state):
            return 'attack'
        return 'defend'

    def enemies_far_away(self, game_state):
        # If an enemy is waiting near the middle, it would get in as soon as
        # we leave, so we only attack if our filter puts every enemy far from
        # our side of the boundary.
        return all(self.distance_home(self.most_likely_position(i)) > 2 * DANGER_DISTANCE
                   for i in self.get_opponents(game_state))
