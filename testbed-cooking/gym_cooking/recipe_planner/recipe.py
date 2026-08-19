from gym_cooking.utils.core import *


class Recipe:
    def __init__(self, name, length, bonus, container=None):
        self.name = name
        self.length = length
        self.bonus = bonus
        # 盛り付ける容器。サラダ/スープは皿、ジュースはコップ。
        # 容器まで含めた文字列が料理の識別子(full_plate_name)になるため、
        # ここを間違えると別の料理と同じ名前になってしまう。
        self.container = container if container is not None else Plate()
        self.contents = []

    def __str__(self):
        return self.name

    def add_ingredient(self, item):
        self.contents.append(item)

    def add_goal(self):
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        self.contents_names = [c.full_name for c in self.contents]   # list of strings
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        container_name = self.container.full_name
        self.full_plate_name = '-'.join(sorted(self.contents_names + [container_name]))   # string
        self.final_task = Object(location=None, contents=self.contents + [type(self.container)()])

class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Tomato', 20, 10)
        self.add_ingredient(Tomato(state_index=2))
        self.add_goal()

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce', 20, 10)
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class SimpleOnion(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Onion', 20, 10)
        self.add_ingredient(Onion(state_index=2))
        self.add_goal()

class TomatoLettuceSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoLettuceSalad', 30, 15)
        self.add_ingredient(Tomato(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class OnionTomatoSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionTomatoSalad', 30, 15)
        self.add_ingredient(Onion(state_index=2))
        self.add_ingredient(Tomato(state_index=2))
        self.add_goal()

class OnionLettuceSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionLettuceSalad', 30, 15)
        self.add_ingredient(Onion(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class FullSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FullSalad', 40, 20)
        self.add_ingredient(Tomato(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_ingredient(Onion(state_index=2))
        self.add_goal()

class OnionSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSoup', 50, 10)
        self.add_ingredient(Onion(state_index=4))
        self.add_goal()

class TomatoSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoSoup', 50, 10)
        self.add_ingredient(Tomato(state_index=4))
        self.add_goal()
        
class LettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'LettuceSoup', 50, 10)
        self.add_ingredient(Lettuce(state_index=4))
        self.add_goal()

class TomatoLettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoLettuceSoup', 60, 15)
        self.add_ingredient(Tomato(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_goal()

class OnionTomatoSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionTomatoSoup', 60, 15)
        self.add_ingredient(Onion(state_index=4))
        self.add_ingredient(Tomato(state_index=4))
        self.add_goal()

class OnionLettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionLettuceSoup', 60, 15)
        self.add_ingredient(Onion(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_goal()

class FullSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FullSoup', 70, 20)
        self.add_ingredient(Tomato(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_ingredient(Onion(state_index=4))
        self.add_goal()



# -----------------------------------------------------------
# ジュース(ミキサーで混ぜてコップに注ぐ)
# 材料の state_index=4 は FRESH_CHOPPING_CHOPPED_MIXING_MIXED の Mixed。
# 容器は皿ではなくコップ。皿にするとサラダと同じ料理名になってしまう。
# -----------------------------------------------------------

class AppleOrangeJuice(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'AppleOrangeJuice', 60, 15, container=Cup())
        self.add_ingredient(Apple(state_index=4))
        self.add_ingredient(Orange(state_index=4))
        self.add_goal()

class AppleBananaJuice(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'AppleBananaJuice', 60, 15, container=Cup())
        self.add_ingredient(Apple(state_index=4))
        self.add_ingredient(Banana(state_index=4))
        self.add_goal()

class BananaOrangeJuice(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'BananaOrangeJuice', 60, 15, container=Cup())
        self.add_ingredient(Banana(state_index=4))
        self.add_ingredient(Orange(state_index=4))
        self.add_goal()
