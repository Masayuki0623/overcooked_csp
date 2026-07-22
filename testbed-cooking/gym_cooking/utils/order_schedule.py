from gym_cooking.utils.core import *
from gym_cooking.utils.config import ORDER_EXPIRE_PUNISH

import numpy as np
import copy
from pathlib import Path
import gym_cooking.recipe_planner.recipe as RECIPE
import gym_cooking


class OrderScheduler:
    def __init__(self, arglist, recipes):
        self.arglist = arglist
        self.disable_order_expiry = True
        self.recipe_name_list = self._load_recipe_name_list(arglist)
        self.recipes = self._resolve_recipes(recipes)
        self.rand_recipe_list = list(range(len(self.recipes)))
        self.rand_recipe_idx = 0

        self.max_num_orders = len(self.recipes)

        self.current_orders = [self.new_order(recipe) for recipe in self.recipes]

        self.reward = 0
        self.successful_orders = 0
        self.failed_orders = 0

    def _load_recipe_name_list(self, arglist):
        order_file = getattr(arglist, 'order_file', None)
        if not order_file:
            return None

        order_path = Path(order_file)
        if order_path.suffix == '':
            order_path = order_path.with_suffix('.txt')
        if not order_path.is_absolute():
            local_path = Path(gym_cooking.__file__).absolute().parent / 'utils' / 'order' / order_file
            if local_path.suffix == '':
                local_path = local_path.with_suffix('.txt')
            if local_path.exists():
                order_path = local_path

        with open(order_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            raise ValueError(f"Order file is empty: {order_path}")

        count = int(lines[0])
        recipe_names = lines[1:1 + count]
        if len(recipe_names) != count:
            raise ValueError(f"Order file count mismatch: expected {count}, got {len(recipe_names)}")
        return recipe_names

    def _resolve_recipes(self, default_recipes):
        if not self.recipe_name_list:
            return default_recipes

        recipes = []
        for recipe_name in self.recipe_name_list:
            if not hasattr(RECIPE, recipe_name):
                raise ValueError(f"Unknown recipe name in order file: {recipe_name}")
            recipes.append(getattr(RECIPE, recipe_name)())
        return recipes

    def assign_rand_recipe_list(self, rand_recipe_list):
        self.rand_recipe_list = rand_recipe_list
        self.rand_recipe_idx = 0

        self.current_orders = []
        for recipe_idx in self.rand_recipe_list[:self.max_num_orders]:
            self.current_orders.append(self.new_order(self.recipes[recipe_idx]))

    def __copy__(self):
        new = OrderScheduler(self.arglist, copy.copy(self.recipes))
        new.recipe_name_list = copy.copy(self.recipe_name_list)
        new.rand_recipe_list = copy.copy(self.rand_recipe_list)
        new.rand_recipe_idx = self.rand_recipe_idx
        new.current_orders = copy.copy(self.current_orders)
        new.reward = self.reward
        new.successful_orders = self.successful_orders
        new.failed_orders = self.failed_orders
        return new

    def new_order(self, recipe):
        goal_obj = copy.deepcopy(recipe.final_task)
        return goal_obj, recipe.length, recipe.length, recipe.bonus

    def update(self, world, passed_time=1.):
        # check completed orders
        current_orders = []
        delivery_list = list(
            filter(lambda o: o.name == 'Delivery', world.get_object_list()))
        for order, restTime, timeLimit, bonus in self.current_orders:
            goal_obj = order
            success = False
            for delivery in delivery_list:
                if delivery.exists(goal_obj.full_name):
                    delivery.pop(goal_obj.full_name)
                    success = True
                    goal_obj.location = delivery.location
                    world.remove(goal_obj)
                    break
            if success:
                self.reward += bonus
                self.successful_orders += 1
            else:
                current_orders.append((order, restTime, timeLimit, bonus))
        self.current_orders = current_orders

        if not self.disable_order_expiry:
            current_orders = []
            for order, restTime, timeLimit, bonus in self.current_orders:
                if restTime - passed_time > 1e-3:
                    current_orders.append(
                        (order, restTime - passed_time, timeLimit, bonus))
                else:
                    self.failed_orders += 1
                    self.reward -= ORDER_EXPIRE_PUNISH
            self.current_orders = current_orders

        # 固定注文列モードでは、新しい注文は生成しない

    def consume_reward(self):
        # temp = self.reward
        # self.reward = 0
        return 0
