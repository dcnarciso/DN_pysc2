from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import math
import numpy as np 
import pandas as pd

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class tbot(base_agent.BaseAgent):
	def __init__(self):
		super(tbot, self).__init__()

		self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
		self.previous_killed_unit_score = 0
		self.previous_killed_building_score = 0

		self.previous_action = None
		self.previous_state = None

		self.attack_coordinates = None

	def unit_type_is_selected(self, obs, unit_type):
		if (len(obs.observation.single_select) > 0 and
				obs.observation.single_select[0].unit_type == unit_type):
			return True

		if (len(obs.observation.multi_select) > 0 and
				obs.observation.multi_select[0].unit_type == unit_type):
			return True

		return False

	def get_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.feature_units 
				if unit.unit_type == unit_type]

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def step(self, obs):
		super(tbot, self).step(obs)

		if obs.first():
			player_y, player_x = (obs.observation.feature_minimap.player_relative ==
								  features.PlayerRelative.SELF).nonzero()
			xmean = player_x.mean()
			ymean = player_y.mean()

			if xmean <= 31 and ymean <= 31:
				self.attack_coordinates = (49, 49)
			else:
				self.attack_coordinates = (12, 16)

########Brain Stuff
		# killed_unit_score = obs.observation.score_cumulative.killed_value_units
		# killed_building_score = obs.observation.score_cumulative.killed_value_structures

		# supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))
		# barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))
		# supply_limit = obs.observation.player.food_cap
		# army_supply = obs.observation.player.food_used - len(self.get_units_by_type(obs, units.Terran.SCV))

		# current_state = [
		# 	supply_depot_count,
		# 	barracks_count,
		# 	supply_limit,
		# 	army_supply,
		# 	]

		# reward = 0

		# if self.previous_action is not None:
		# 	reward = 0
		# 	if killed_unit_score > self.previous_killed_unit_score:
		# 		reward += KILL_UNIT_REWARD
		# 	if killed_building_score > self.previous_killed_building_score:
		# 		reward += KILL_BUILDING_REWARD

		# 	self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

		# rl_action = self.qlearn.choose_action(str(current_state))
		# smart_action = smart_actions[rl_action]

		# self.previous_killed_unit_score = killed_unit_score
		# self.previous_killed_building_score = killed_building_score
		# self.previous_state = current_state
		# self.previous_action = rl_action

		# if smart_action == ACTION_DO_NOTHING:
		# 	return actions.FUNCTIONS.no_op()

		# elif smart_action == ACTION_SELECT_SCV:
		# 	scvs = self.get_units_by_type(obs, units.Terran.SCV)
		# 	if len(scvs) > 0:
		# 		scv = random.choice(scvs)
		# 		return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
		# 																  scv.y))
		# elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
		# 	depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
		# 	if len(depots) == 0:
		# 		if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 			if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in
		# 				obs.observation.available_actions):
		# 				x = random.randint(0, 83)
		# 				y = random.randint(0, 83)

		# 				return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))

		# 	if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 		free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
		# 		if free_supply <= 2:
		# 			if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
		# 				x = random.randint(0, 83)
		# 				y = random.randint(0, 83)
		# 				return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x,y))	

		# elif smart_action == ACTION_BUILD_BARRACKS:
		# 	scvs = self.get_units_by_type(obs, units.Terran.SCV)
		# 	if len(scvs) > 0:
		# 		scv = random.choice(scvs)
		# 		return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
		# 																  scv.y))

		# 	if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 		if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
		# 			x = random.randint(0, 83)
		# 			y = random.randint(0, 83)
		# 			return actions.FUNCTIONS.Build_Barracks_screen("now", (x,y))


		# elif smart_action == ACTION_SELECT_BARRACKS:
		# 	barrackses = self.get_units_by_type(obs, units.Terran.Barracks)
		# 	if len(barrackses) > 0:
		# 		barracks = random.choice(barrackses)
		# 		return actions.FUNCTIONS.select_point("select_all_type", (barracks.x,
		# 																	barracks.y))
		# elif smart_action == ACTION_BUILD_MARINE:
		# 	if self.unit_type_is_selected(obs, units.Terran.Barracks):
		# 		free_supply = (obs.observation.player.food_cap -
		# 						obs.observation.player.food_used)
		# 		if free_supply >= 0 :
		# 			if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
		# 				return actions.FUNCTIONS.Train_Marine_quick("now")

		# elif smart_action == ACTION_SELECT_ARMY:
		# 	if self.can_do(obs, actions.FUNCTIONS.select_army.id):
		# 		return actions.FUNCTIONS.select_army("select")

		# elif smart_action == ACTION_ATTACK:
		# 	if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
		# 		return actions.FUNCTIONS.Attack_minimap("now",
		# 												self.attack_coordinates)

########Brain Stuff
		# if obs.first():
		# 	player_y, player_x = (obs.observation.feature_minimap.player_relative ==
		# 						  features.PlayerRelative.SELF).nonzero()
		# 	xmean = player_x.mean()
		# 	ymean = player_y.mean()

		# 	if xmean <= 31 and ymean <= 31:
		# 		self.attack_coordinates = (49, 49)
		# 	else:
		# 		self.attack_coordinates = (12, 16)

  #       self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0


		# marines = self.get_units_by_type(obs, units.Terran.Marine)
		# if len(marines) >= 10:
		# 	if self.unit_type_is_selected(obs, units.Terran.Marine):
		# 		if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
		# 			return actions.FUNCTIONS.Attack_minimap("now",
		# 													self.attack_coordinates)

		# 	if self.can_do(obs, actions.FUNCTIONS.select_army.id):
		# 		return actions.FUNCTIONS.select_army("select")

		# depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
		# if len(depots) == 0:
		# 	if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 		if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in
		# 			obs.observation.available_actions):
		# 			x = random.randint(0, 83)
		# 			y = random.randint(0, 83)

		# 			return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))

		# if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 	free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
		# 	if free_supply <= 2:
		# 		if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
		# 			x = random.randint(0, 83)
		# 			y = random.randint(0, 83)
		# 			return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x,y))	

		# if self.unit_type_is_selected(obs, units.Terran.Barracks):
		# 	free_supply = (obs.observation.player.food_cap -
		# 					obs.observation.player.food_used)
		# 	if free_supply >= 0 :
		# 		if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
		# 			return actions.FUNCTIONS.Train_Marine_quick("now")

		# barrackses = self.get_units_by_type(obs, units.Terran.Barracks)
		# if len(barrackses) < 4:
		# 	if self.unit_type_is_selected(obs, units.Terran.SCV):
		# 		if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
		# 			x = random.randint(0, 83)
		# 			y = random.randint(0, 83)
		# 			return actions.FUNCTIONS.Build_Barracks_screen("now", (x,y))

		# 	scvs = self.get_units_by_type(obs, units.Terran.SCV)
		# 	if len(scvs) > 0:
		# 		scv = random.choice(scvs)
		# 		return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
		# 																  scv.y))

		# ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
		# if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
		# 	if self.can_do(obs, actions.FUNCTIONS.Train_SCV_quick.id):
		# 		return actions.FUNCTIONS.Train_SCV_quick("now")


		# return actions.FUNCTIONS.no_op()


def main(unused_argv):
	agent = tbot()
	try:
		while True:
			with sc2_env.SC2Env(
				map_name = "AbyssalReef",
				players = [sc2_env.Agent(sc2_env.Race.terran),
						   sc2_env.Bot(sc2_env.Race.random,
						   			   sc2_env.Difficulty.very_easy)],
				agent_interface_format = features.AgentInterfaceFormat(
					feature_dimensions = features.Dimensions(screen=84, minimap = 64),
					use_feature_units = True),
				step_mul = 16,
				game_steps_per_episode = 0,
				visualize = True) as env:

				agent.setup(env.observation_spec(), env.action_spec())

				timesteps = env.reset()
				agent.reset()

				while True:
					step_actions = [agent.step(timesteps[0])]
					if timesteps[0].last():
						break
					timesteps = env.step(step_actions)
	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
	app.run(main)