import random
import gym
from gym import spaces
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from copy import copy

RIGHT = 0
LEFT = 1
DOWN = 3
UP = 2

x_size = 11
y_size = 11

EMPTY = 0
TAIL = 1
HEAD = 2
APPLE = 3

class SnakeEnv(gym.Env):
    """
    Snake Game

    Description:
        OpenAI Gym Environment for the Classic Snake Game

    Action Space:
        Discrete(4)
        RIGHT = 0; LEFT = 1; DOWN = 3; UP = 2
    Observation Space:
        12x12 matrix of cells marked for empty cells = 0, snake tail = 1, snake head = 2, apple = 3
    
    Step:
        Moves the snake


    """
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 4
    }

    def __init__(self):
        self.x_min = 0
        self.x_max = x_size
        self.y_min = 0
        self.y_max = y_size
        self.viewer = None
        

        self.score = 0

        self.head_x = 5
        self.head_y = 5
        self.body = [(4, 5), (3, 5)]

        valid_cells = _get_valid_cells((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.body)
        
        self.apple_x, self.apple_y = valid_cells[random.randint(0, len(valid_cells)-1)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = 0, high = 3, shape = (self.x_max-1, self.y_max-1), dtype = np.float64)

        self.state = _get_gamestate((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.apple_x, self.apple_y, self.body)
        
        #for rendering
        self.init_ax, self.init_ay = copy(self.apple_x), copy(self.apple_y)
        self.init_hx, self.init_hy = copy(self.head_x), copy(self.head_y)
        self.init_body = copy(self.body)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        #moves head and adds to body
        self.body = [(self.head_x, self.head_y)] + self.body
        if action == RIGHT:
            self.head_x += 1
        elif action == LEFT:
            self.head_x -= 1
        elif action == UP:
            self.head_y += 1
        elif action == DOWN:
            self.head_y -= 1

        #score function
        self.score -= 1
        done = False
        
        info = {}

        #check for collision
        if self.head_x == self.x_min or self.head_x == self.x_max or self.head_y == self.y_min or self.head_y == self.y_max:
            self.score -= 100
            done = True
            return self.state, self.score, done, info

        elif (self.head_x, self.head_y) in self.body:
            self.score -= 100
            done = True
            return self.state, self.score, done, info
        
        if self.head_x == self.apple_x and self.head_y == self.apple_y:
            self.score += 100
            valid_cells = _get_valid_cells((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.body)
            self.apple_x, self.apple_y = valid_cells[random.randint(0, len(valid_cells)-1)]
            self.init_body.append(self.body[-1])

        else:
            #snake does not grow if it did not eat the apple
            del self.body[-1]

        self.state = _get_gamestate((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.apple_x, self.apple_y, self.body)

        return self.state, self.score, done, info

    def reset(self):
        self.close()
        self.score = 0
        self.head_x = 5
        self.head_y = 5

        self.body = [(4, 5), (3, 5)]

        valid_cells = _get_valid_cells((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.body)
        self.apple_x, self.apple_y = valid_cells[random.randint(0, len(valid_cells)-1)]
        
        self.state = _get_gamestate((self.x_max - 1, self.y_max - 1), self.head_x, self.head_y, self.apple_x, self.apple_y, self.body)
        
        self.init_ax, self.init_ay = copy(self.apple_x), copy(self.apple_y)
        self.init_hx, self.init_hy = copy(self.head_x), copy(self.head_y)
        self.init_body = copy(self.body)
        
        return self.state
    
    def render(self, mode = 'human', close = False):
        from gym.envs.classic_control import rendering
        screen_width = 400
        screen_height = 400
        
        scale_x = screen_width/(self.x_max - 1)
        scale_y= screen_height/(self.y_max - 1)
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            #plots the apple
            l, r, t, b = (self.apple_x-1) * scale_x, (self.apple_x) * scale_x, (self.apple_y-1) * scale_y, (self.apple_y) * scale_y
            apple = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            apple.set_color(0.9, 0.1, 0.1)
            self.appletrans = rendering.Transform()
            apple.add_attr(self.appletrans)
            self.viewer.add_geom(apple)
            
            #plots the head
            l, r, t, b = (self.head_x-1) * scale_x, (self.head_x) * scale_x, (self.head_y-1) * scale_y, (self.head_y) * scale_y
            head = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            head.set_color(0.1, 0.9, 0.1)
            self.headtrans = rendering.Transform()
            head.add_attr(self.headtrans)
            self.viewer.add_geom(head)
            
            #plots the body
            for i, (x, y) in enumerate(self.body):
                l, r, t, b = (x-1) * scale_x, (x) * scale_x, (y-1) * scale_y, (y) * scale_y
                exec(f"body{i} = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])")
                exec(f"body{i}.set_color(0.3, 0.8, 0.3)")
                exec(f"self.transbody{i} = rendering.Transform()")
                exec(f"body{i}.add_attr(self.transbody{i})")
                exec(f"self.viewer.add_geom(body{i})")
        
        #moves things around
        a_x, a_y = (self.apple_x - self.init_ax) * scale_x, (self.apple_y - self.init_ay) * scale_y
        h_x, h_y = (self.head_x - self.init_hx) * scale_x, (self.head_y - self.init_hy) * scale_y
        self.appletrans.set_translation(a_x, a_y)
        self.headtrans.set_translation(h_x, h_y)

        for t, (x, y) in enumerate(self.body):
            try: 
                temp_x, temp_y = (x - self.init_body[t][0]) * scale_x, (y - self.init_body[t][1]) * scale_y
                exec(f"self.transbody{t}.set_translation({temp_x}, {temp_y})")
            except:
                i = len(self.body) - 1
                (x,y) = self.body[i]
                l, r, t, b = (x-1) * scale_x, (x) * scale_x, (y-1) * scale_y, (y) * scale_y
                exec(f"body{i} = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])")
                exec(f"body{i}.set_color(0.3, 0.8, 0.3)")
                exec(f"self.transbody{i} = rendering.Transform()")
                exec(f"body{i}.add_attr(self.transbody{i})")
                exec(f"self.viewer.add_geom(body{i})")
                temp_x, temp_y = (x - self.init_body[t][0]) * scale_x, (y - self.init_body[t][1]) * scale_y
                exec(f"self.transbody{t}.set_translation({temp_x}, {temp_y})")
                
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def _get_gamestate(max_tuple, head_x, head_y, apple_x, apple_y, body):
    #Calculates the game state as an array
    array = np.ones(max_tuple)*EMPTY
    array[apple_y-1, apple_x-1] = APPLE
    for (x, y) in body:
        array[y-1, x-1] = TAIL
    array[head_y-1, head_x-1] = HEAD
    return array

def _get_valid_cells(max_tuple, head_x, head_y, body):
    #gets the valid cells for an apple
    xs = range(1, max_tuple[0])
    ys = range(1, max_tuple[1])
    cells = product(xs, ys)
    return ([i for i in cells if i not in ([(head_x, head_y)] + body)])
