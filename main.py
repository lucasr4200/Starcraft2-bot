import math
import random
import numpy as np
import pandas as pd
import os

import pysc2
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
import pickle


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):  # e_greedy = 0.9
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)

        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        # print("!!!!!!", self.q_table.loc[s, a])
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat(
                [self.q_table, pd.DataFrame([[0] * len(self.actions)], columns=self.actions, index=[state])])


class Agent(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "lower_supply_depot",
               "build_barracks",
               "build_barracks_techlab",
               "build_refinery",
               "train_marine",
               "train_scv",
               "train_marauder",
               "attack")

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if
                unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.SELF]

    def get_neutral_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if
                unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.NEUTRAL]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if
                unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if
                unit.unit_type == unit_type and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if
                unit.unit_type == unit_type and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        if xy is None:
            return 4
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(supply_depots) < 6 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            build_location = self.get_build_location(obs)
            if build_location is not None:
                distances = self.get_distances(obs, scvs, build_location)
                scv = scvs[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                    "now", scv.tag, build_location)
        return actions.RAW_FUNCTIONS.no_op()

    def get_build_location(self, obs):
        if self.get_my_units_by_type(obs, units.Terran.CommandCenter):
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            if len(command_center) == 0:
                return None
            base_location = (command_center.x, command_center.y)
            x = base_location[0] + random.randint(-5, 5)
            y = base_location[1] + random.randint(-4, 4)
            if not self.base_top_left:
                x = 83 - x
                y = 83 - y
            if (x, y) is None:
                # self.get_build_location(obs)
                self.do_nothing(obs)
            else:
                return (x, y)

    def building_exists(self, obs, x, y):
        """
        Check if a building of the specified type exists at the given (x, y) location.
        """
        for unit in obs.observation.raw_units:
            if abs(unit.x - x) < 2 and abs(unit.y - y) < 2:
                return True
        return False

    def lower_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        if len(supply_depots) > 0:
            supply_depot = random.choice(supply_depots)
            return actions.RAW_FUNCTIONS.Morph_SupplyDepot_Lower_quick("now", supply_depot.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) < 3 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            build_location = self.get_build_location(obs)

            if build_location is None:
                return actions.RAW_FUNCTIONS.no_op()

            # barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, build_location)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, build_location)
        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        if (len(completed_barrackses) > 0 and len(refineries) == 0 and
                obs.observation.player.minerals >= 75 and len(scvs) > 0):
            geysers = self.get_neutral_units_by_type(obs, units.Neutral.VespeneGeyser)
            if command_centers:
                cc = command_centers[0]
                distances = [self.distance_between_units(cc, geyser) for geyser in geysers]
                closest_geyser = geysers[np.argmin(distances)]
                scv = scvs[0]

                return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                    "now", scv.tag, closest_geyser.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def distance_between_units(self, unit1, unit2):
        return ((unit1.x - unit2.x) ** 2 + (unit1.y - unit2.y) ** 2) ** 0.5

    def build_barracks_techlab(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        techlabs = self.get_my_units_by_type(obs, units.Terran.TechLab)
        # barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)
        if (len(completed_barrackses) > 0 and len(techlabs) == 0 and
                obs.observation.player.minerals >= 75):
            techlab_xy = (22, 21) if self.base_top_left else (35, 45)
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick(
                "now", barracks.tag)  # , techlab_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marauder(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        completed_command_centers = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_command_centers) > 0 and obs.observation.player.minerals >= 50
                and free_supply > 0 and len(self.get_my_units_by_type(obs, units.Terran.SCV)) < 20):
            command_center = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            if command_center.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)
        units_to_attack = marines + marauders
        if units_to_attack:
            attack_xy = (39, 48) if self.base_top_left else (18, 20)
            distances = self.get_distances(obs, units_to_attack, attack_xy)
            unit = units_to_attack[np.argmax(distances)]
            x_offset = random.randint(-6, 6)
            y_offset = random.randint(-6, 6)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", unit.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.actions)
        # try:
        #     with open('q_table.pkl', 'rb') as f:
        #         self.qtable = pickle.load(f)
        # except Exception as e:
        #     print(f'Error loading Q-table: {e}')
        self.new_game()

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        queued_marauders = (completed_barrackses[0].order_length
                            if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        can_afford_techlab = obs.observation.player.minerals >= 50
        can_afford_marauder = obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 25
        can_afford_scv = obs.observation.player.minerals >= 50
        can_afford_refinery = obs.observation.player.minerals >= 75

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                len(marauders),
                len(refineries),
                queued_marines,
                queued_marauders,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_techlab,
                can_afford_marine,
                can_afford_marauder,
                can_afford_scv,
                can_afford_refinery,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        if self.episodes % 10 == 0:
            try:
                with open('q_table.pkl', 'wb') as f:
                    pickle.dump(self.qtable, f)
            except Exception as e:
                print(f'Error saving Q-table: {e}')

        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward,
                              'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)


def main(unused_argv):
    agent1 = SmartAgent()
    agent2 = RandomAgent()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=80,
                disable_fog=True,
                realtime=False,
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=10000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
