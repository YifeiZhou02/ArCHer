# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Dict
import jericho

import textworld
from textworld.core import EnvInfos, GameState
from textworld.generator.game import GameProgression
from textworld.generator.inform7 import Inform7Game
# from LLM_RL.environment import Text, TextEnv, TextHistory
import os
import random

DEFAULT_OBSERVATION = """
[To get text observation use the '.z8' or '.ulx' files instead of the '.json' one.]
# """
class JerichoEnv(textworld.Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seed = -1
        self._jericho = None
        self.gamefile = None
        self._reset = False

    def load(self, z_file: str) -> None:
        self.gamefile = os.path.abspath(z_file)
        _, ext = os.path.splitext(os.path.basename(self.gamefile))

        # Check if game is supported by Jericho.
        if not ext.startswith(".z"):
            raise ValueError("Only .z[1-8] files are supported!")

        if not os.path.isfile(self.gamefile):
            raise FileNotFoundError(self.gamefile)

        if self._jericho is None:
            # Start the game using Jericho.
            self._jericho = jericho.FrotzEnv(self.gamefile, self._seed)
        else:
            self._jericho.load(self.gamefile)

    def __del__(self) -> None:
        self.close()

    @property
    def game_running(self) -> bool:
        """ Determines if the game is still running. """
        return self._jericho is not None

    def seed(self, seed=None):
        self._seed = seed
        if self._jericho:
            self._jericho.seed(self._seed)

        return self._seed

    def _gather_infos_jenv(self):
        # print("Gathering info jenv")
        """ Adds additional information to the internal state. """
        self.state.feedback = self.state.raw
        if not self._jericho.is_fully_supported:
            return  # No more information can be gathered.

        for attr in self.infos.basics:
            self.state[attr] = getattr(self._jericho, "get_" + attr, lambda: self.state.get(attr))()

        for attr in self.infos.extras:
            self.state["extra.{}".format(attr)] = getattr(self._jericho, "get_" + attr, lambda: None)()

        # Deal with information that has different method name in Jericho.
        self.state["won"] = self._jericho.victory()
        self.state["lost"] = self._jericho.game_over()
        self.state["score"] = self._jericho.get_score()
        self.state["moves"] = self._jericho.get_moves()
        self.state["location"] = self._jericho.get_player_location()

        bkp = self._jericho.get_state()
        self.state["description"], _, _, _ = self._jericho.step("look")
        self._jericho.set_state(bkp)

        if self.infos.inventory:
            bkp = self._jericho.get_state()
            self.state["inventory"], _, _, _ = self._jericho.step("inventory")
            self._jericho.set_state(bkp)

        if self.infos.admissible_commands:
            self.state["_valid_commands"] = self._jericho.get_valid_actions()
            self.state["admissible_commands"] = sorted(set(self.state["_valid_commands"]))

    def reset(self):
        if not self.game_running:
            raise GameNotRunningError("Call env.load(gamefile) before env.reset().")

        self.state = GameState()
        self.state.raw, _ = self._jericho.reset()
        self._gather_infos_jenv()
        self._reset = True
        return self.state

    def _send(self, command: str) -> str:
        """ Send a command directly to the interpreter.

        This method will not affect the internal state variable.
        """
        feedback, _, _, _ = self._jericho.step(command)
        return feedback

    def step(self, command):
        if not self.game_running or not self._reset:
            raise GameNotRunningError()

        self.state = GameState()
        self.state.last_command = command.strip()
        res = self._jericho.step(self.state.last_command)
        # As of Jericho >= 2.1.0, the reward is returned instead of the score.
        self.state.raw, _, self.state.done, _ = res
        self._gather_infos_jenv()
        return self.state, self.state.score, self.state.done

    def close(self):
        if self.game_running:
            self._jericho.close()
            self._jericho = None
            self._reset = False

    def copy(self) -> "JerichoEnv":
        """ Return a copy of this environment at the same state. """
        env = JerichoEnv(self.infos)
        env._seed = self._seed

        if self.gamefile:
            env.load(self.gamefile)

        if self._jericho:
            env._jericho = self._jericho.copy()
            env._reset = True

        # Copy core Environment's attributes.
        env.state = self.state.copy()
        env.infos = self.infos.copy()
        return env


class TextWorldEnv(textworld.Environment):
    """
    Environment for playing games by TextWorld.
    """

    def __init__(self, infos: Optional[EnvInfos] = None) -> None:
        """
        Arguments:
            infos: Information to be included in the game state. By
                   default, only the game's narrative is included.
        """
        super().__init__(infos)
        self._gamefile = None
        self._game = None
        self._inform7 = None
        self._last_action = None
        self._prev_state = None
        self._previous_winning_policy = None
        self._current_winning_policy = None
        self._moves = None
        self._game_progression = None

    def load(self, path: str) -> None:
        self._gamefile = path
        self._game = textworld.Game.load(self._gamefile)
        self._game_progression = None
        self._inform7 = Inform7Game(self._game)

    def _gather_infos(self):
        self.state["game"] = self._game
        self.state["command_templates"] = self._game.command_templates
        self.state["verbs"] = self._game.verbs
        self.state["entities"] = self._game.entity_names
        self.state["objective"] = self._game.objective
        self.state["max_score"] = self._game.max_score

        for k, v in self._game.metadata.items():
            self.state["extra.{}".format(k)] = v

        self.state["_game_progression"] = self._game_progression
        self.state["_facts"] = list(self._game_progression.state.facts)

        self.state["won"] = self._game_progression.completed
        self.state["lost"] = self._game_progression.failed

        self.state["_winning_policy"] = self._current_winning_policy
        if self.infos.policy_commands:
            self.state["policy_commands"] = []
            if self._game_progression.winning_policy is not None:
                self.state["policy_commands"] = self._inform7.gen_commands_from_actions(self._current_winning_policy)

        if True:
            self.state["intermediate_reward"] = 0
            if self.state["won"]:
                # The last action led to winning the game.
                self.state["intermediate_reward"] = 1

            elif self.state["lost"]:
                # The last action led to losing the game.
                self.state["intermediate_reward"] = -1

            elif self._previous_winning_policy is None:
                self.state["intermediate_reward"] = 0

            else:
                diff = len(self._previous_winning_policy) - len(self._current_winning_policy)
                self.state["intermediate_reward"] = int(diff > 0) - int(diff < 0)  # Sign function.

        if self.infos.facts:
            self.state["facts"] = list(map(self._inform7.get_human_readable_fact, self.state["_facts"]))

        self.state["last_action"] = None
        self.state["_last_action"] = self._last_action
        if self.infos.last_action and self._last_action is not None:
            self.state["last_action"] = self._inform7.get_human_readable_action(self._last_action)

        self.state["_valid_actions"] = self._game_progression.valid_actions
        self.state["_valid_commands"] = self._inform7.gen_commands_from_actions(self._game_progression.valid_actions)
        # To guarantee the order from one execution to another, we sort the commands.
        # Remove any potential duplicate commands (they would lead to the same result anyway).
        self.state["admissible_commands"] = sorted(set(self.state["_valid_commands"]))
        redundant = ["examine", "look", "inventory"]
        self.state["admissible_commands"] = list(
                c for c in self.state["admissible_commands"] if not any(a in c for a in redundant))
        if self.infos.moves:
            self.state["moves"] = self._moves

    def reset(self):
        self._prev_state = None
        self.state = GameState()
        self._game_progression = GameProgression(self._game, track_quests=True)
        self._last_action = None
        self._previous_winning_policy = None
        self._current_winning_policy = self._game_progression.winning_policy
        self._moves = 0

        self.state.raw = DEFAULT_OBSERVATION
        self.state.feedback = DEFAULT_OBSERVATION
        self._gather_infos()
        return self.state

    def step(self, command: str):
        command = command.strip()
        self._prev_state = self.state

        self.state = GameState()
        self.state.last_command = command
        self.state.raw = DEFAULT_OBSERVATION
        self.state.feedback = DEFAULT_OBSERVATION
        self._previous_winning_policy = self._current_winning_policy

        self._last_action = None
        try:
            # Find the action corresponding to the command.
            idx = self._prev_state["_valid_commands"].index(command)
            self._last_action = self._game_progression.valid_actions[idx]
            # An action that affects the state of the game.
            self._game_progression.update(self._last_action)
            self._current_winning_policy = self._game_progression.winning_policy
            self._moves += 1
        except ValueError:
            self.state.feedback = "Invalid command."
            pass  # We assume nothing happened in the game.

        self._gather_infos()
        self.state["score"] = self._game_progression.score
        self.state["done"] = self.state["won"] or self.state["lost"]
        return self.state, self.state["score"], self.state["done"]

    def copy(self) -> "TextWorldEnv":
        """ Return a copy of this environment.

        It is safe to call `step` and `reset` on the copied environment.

        .. warning:: The `Game` and `Inform7Game` private objects are *soft* copies.
        """
        env = TextWorldEnv()

        # Copy core Environment's attributes.
        env.state = self.state.copy()
        env.infos = self.infos.copy()

        env._gamefile = self._gamefile
        env._game = self._game  # Reference
        env._inform7 = self._inform7  # Reference

        env._prev_state = self._prev_state.copy() if self._prev_state is not None else None
        env._last_action = self._last_action
        env._moves = self._moves
        if self._previous_winning_policy is not None:
            env._previous_winning_policy = tuple(self._previous_winning_policy)

        if self._current_winning_policy is not None:
            env._current_winning_policy = tuple(self._current_winning_policy)

        if self._game_progression is not None:
            env._game_progression = self._game_progression.copy()

        return env


# An environment that contains more information for our purpose
class TextNavEnv():
    """
    Environment for playing games by TextWorld.
    """

    def __init__(self, max_env_steps=10) -> None:
        """
        Arguments:
            infos: Information to be included in the game state. By
                   default, only the game's narrative is included.
        """
        super().__init__()
        # TestWorldEnv has everything else
        self.t_env = TextWorldEnv()
        #only JerichoEnv has text observations
        self.j_env = JerichoEnv()
        self.state = {}
        self.objective = None
        self.walkthrough = None
        self.done = False
        self.max_env_steps = max_env_steps

    def load(self, path: str) -> None:
        self.t_env.load(path+ '.json')
        self.j_env.load(path+ '.z8')
        self.objective = self.t_env._game.objective
    
    ## TODO: remove providing available options if possible
    def get_observation(self, feedback:str):
        bkp = self.j_env._jericho.get_state()
        description, _, _, _ = self.j_env._jericho.step("look")
        self.j_env._jericho.set_state(bkp)
        bkp = self.j_env._jericho.get_state()
        inventory, _, _, _ = self.j_env._jericho.step("inventory")
        self.j_env._jericho.set_state(bkp)
        instruction = "You are now playing a exciting episode of TextWorld! Here is your final goal for today."
        last_objective = '.'.join(self.objective.split('.')[-2:])
        # return feedback
        return f"Steps: {str(self.steps)}" + instruction + '\n' + last_objective + "\nState Description:" \
            + description + inventory 
            # +"\nEnvironment Feedback:" +feedback 
            # + '\nAvailable Options: '+ str(self.state['admissible_commands'])\
            # + "Your choice: \n\n"

    def _reset(self):
        self.state = self.t_env.reset()
        j_state = self.j_env.reset()
        self.state['feedback'] = j_state['feedback'].split('\n\n>')[0]
        self.state['feedback'] = self.state['feedback'].split('-=')[1]
        # self.state['description'] = j_state['description']
        self.walkthrough = self.state['extra.walkthrough']
        self.done = False
        self.steps = 0
        return self.get_observation(self.state['feedback'])

    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None):
        obs = self._reset()
        return obs
    
    def _step(self, command:str):
        self.steps += 1
        if self.done:
            raise Exception('The environment is already over')
        new_state, r, done = self.t_env.step(command)
        if self.steps >= self.max_env_steps:
            self.done = True
        #only step if t env can interpret the command
        #TODO: the other option is to only step the jericho env
        if new_state['feedback'] == 'Invalid command.':
            return self.get_observation('Invalid command.'), -1, self.done
        else:
            j_state, _, _ = self.j_env.step(command)
        self.state = new_state
        if not self.done:
            self.done = done
        self.state['feedback'] = j_state['feedback'].split('\n\n>')[0]
        return self.get_observation(self.state['feedback']), self.state['intermediate_reward'], self.done

    def step(self, command):
        command = command.split('<end>')[0]
        return self._step(command)

    def copy(self):
        nenv = TextNavEnv(self.max_env_steps)
        nenv.objective = self.objective
        nenv.done = self.done
        nenv.t_env = self.t_env.copy()
        nenv.j_env = self.j_env.copy()
        return nenv 

    

# An environment that contains more information for our purpose
class ContextualTextNavEnv():
    """
    Environment for playing games by TextWorld.
    """

    def __init__(self, max_env_steps=5) -> None:
        """
        Arguments:
            infos: Information to be included in the game state. By
                   default, only the game's narrative is included.
        """
        super().__init__()
        # print("maximum steps: "+ str(max_env_steps))
        self.state = {}
        self.objective = None
        self.walkthrough = None
        self.done = False
        self.max_env_steps = max_env_steps
        self.context_env = []
        self.current_env = None

    def load(self, cutoff = 100, base_path='./simple_game/'):
        assert len(self.context_env)==0, "can only load once"
        for i in range(cutoff):
            env = TextNavEnv(max_env_steps=self.max_env_steps)
            env.load(base_path + str(i))
            self.context_env.append(env)
        return
    
    def _gather_info(self):
        self.objective = self.current_env.objective
        self.done = self.current_env.done
        self.walkthrough = self.current_env.walkthrough
    
    def reset(self, idx: Optional[int]=None):
        if idx is None:
            self.current_env = random.choice(self.context_env)
        else:
            self.current_env = self.context_env[idx]
        obs = self.current_env.reset()
        self._gather_info()
        return f'Env Index: {str(self.context_env.index(self.current_env))}'+obs
    
    def step(self, text_history):
        result = self.current_env.step(text_history)
        self._gather_info()
        return result

    def copy(self):
        nenv = ContextualTextNavEnv(self.max_env_steps)
        nenv.objective = self.objective
        nenv.done = self.done
        nenv.walkthrough = self.walkthrough
        nenv.context_env = [env.copy() for env in self.context_env]
        return nenv
