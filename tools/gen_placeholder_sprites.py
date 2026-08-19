"""フルーツ/コップ/ミキサーの仮スプライトを生成する。

本番の絵ができたら、同じファイル名で
`testbed-cooking/gym_cooking/misc/game/graphics/` に上書きすれば差し替わる。
このスクリプトは既存ファイルを上書きしないので、差し替え後に再実行しても
本番の絵は消えない(--force を付けたときだけ上書きする)。

    python tools/gen_placeholder_sprites.py

描画は `Game.draw` が `graphics/<full_name>.png` を読むだけなので、
料理名の組み合わせぶんだけ画像が要る。名前の一覧は core.py の
ASSEMBLE_* から取るため、食材を足してもこのスクリプトを流し直せば揃う。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'testbed-cooking'))

import pygame  # noqa: E402

import gym_cooking  # noqa: E402
from gym_cooking.utils.core import (  # noqa: E402
    FRUITS,
    ASSEMBLE_CHOPPED_FRUIT,
    ASSEMBLE_MIXING_FOOD,
    ASSEMBLE_MIXED_FOOD,
    ASSEMBLE_MIXED_CUP_FOOD,
    ASSEMBLE_MIXING_CUP_FOOD,
)

GRAPHICS = Path(gym_cooking.__file__).absolute().parent / 'misc' / 'game' / 'graphics'
SIZE = 40

# 仮の色。本番の絵に差し替えるまでの見分けがつけば十分。
COLORS = {
    'Apple':  (209, 53, 43),
    'Orange': (232, 135, 26),
    'Banana': (232, 201, 58),
}
CUP_COLOR = (236, 240, 245)
BLENDER_COLOR = (150, 160, 175)


def _ingredients_of(full_name):
    """'ChoppedApple-ChoppedBanana' -> ['Apple', 'Banana']"""
    out = []
    for part in full_name.split('-'):
        for fruit in FRUITS:
            if part.endswith(fruit):
                out.append(fruit)
                break
    return out


def _blend(colors):
    if not colors:
        return (128, 128, 128)
    n = len(colors)
    return tuple(sum(c[i] for c in colors) // n for i in range(3))


def _surface():
    s = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))
    return s


def draw_fresh(fruit):
    s = _surface()
    pygame.draw.circle(s, COLORS[fruit], (SIZE // 2, SIZE // 2), SIZE // 2 - 3)
    pygame.draw.circle(s, (40, 40, 40), (SIZE // 2, SIZE // 2), SIZE // 2 - 3, 2)
    return s


def draw_chopped(full_name):
    """刻んだ状態。含まれる食材の数だけ縦帯に分けて塗る。"""
    ings = _ingredients_of(full_name)
    s = _surface()
    if not ings:
        return s
    w = SIZE // len(ings)
    for i, ing in enumerate(ings):
        # 刻んであることが分かるよう、上下2つの矩形に割る
        for dy in (0, SIZE // 2):
            r = pygame.Rect(i * w + 2, dy + 4, w - 4, SIZE // 2 - 8)
            pygame.draw.rect(s, COLORS[ing], r, border_radius=3)
            pygame.draw.rect(s, (40, 40, 40), r, 1, border_radius=3)
    return s


def draw_mixed(full_name, mixing=False):
    """混ざった状態。materials の色を平均した1色の塊にする。"""
    ings = _ingredients_of(full_name)
    s = _surface()
    col = _blend([COLORS[i] for i in ings])
    r = pygame.Rect(4, 6, SIZE - 8, SIZE - 12)
    pygame.draw.rect(s, col, r, border_radius=6)
    pygame.draw.rect(s, (40, 40, 40), r, 2, border_radius=6)
    if mixing:
        # 混ぜている途中は渦を描いて区別する
        pygame.draw.arc(s, (255, 255, 255), r.inflate(-10, -12), 0.4, 3.6, 3)
    return s


def draw_cup():
    s = _surface()
    body = pygame.Rect(9, 8, SIZE - 18, SIZE - 14)
    pygame.draw.rect(s, CUP_COLOR, body, border_radius=4)
    pygame.draw.rect(s, (60, 60, 60), body, 2, border_radius=4)
    return s


def draw_cup_with(full_name):
    s = draw_cup()
    ings = _ingredients_of(full_name)
    if ings:
        col = _blend([COLORS[i] for i in ings])
        inner = pygame.Rect(12, 14, SIZE - 24, SIZE - 24)
        pygame.draw.rect(s, col, inner, border_radius=3)
    return s


def draw_tile(inner):
    """食材の供給台。既存の FreshTomatoTile と同じく、台の上に食材を描く。"""
    s = _surface()
    s.fill((196, 164, 132))
    pygame.draw.rect(s, (120, 96, 72), pygame.Rect(0, 0, SIZE, SIZE), 2)
    small = pygame.transform.smoothscale(inner, (SIZE - 10, SIZE - 10))
    s.blit(small, (5, 5))
    return s


def draw_blender():
    s = _surface()
    s.fill(BLENDER_COLOR)
    pygame.draw.rect(s, (70, 78, 90), pygame.Rect(0, 0, SIZE, SIZE), 2)
    jar = pygame.Rect(11, 6, SIZE - 22, SIZE - 16)
    pygame.draw.rect(s, (225, 232, 240), jar, border_radius=3)
    pygame.draw.rect(s, (70, 78, 90), jar, 2, border_radius=3)
    pygame.draw.rect(s, (70, 78, 90), pygame.Rect(9, SIZE - 10, SIZE - 18, 6),
                     border_radius=2)
    return s


def build():
    """ファイル名 -> Surface。描画側が要求しうる名前をすべて用意する。"""
    out = {}

    for fruit in FRUITS:
        fresh = draw_fresh(fruit)
        out[f'Fresh{fruit}'] = fresh
        out[f'Fresh{fruit}Tile'] = draw_tile(fresh)
        # 刻んでいる途中のコマ(既存の野菜と同じ 1..3 の3枚)
        for i in (1, 2, 3):
            out[f'Chopping{i}{fruit}'] = draw_chopped(f'Chopped{fruit}')

    for name in ASSEMBLE_CHOPPED_FRUIT:
        out[name] = draw_chopped(name)
    for name in ASSEMBLE_MIXING_FOOD:
        out[name] = draw_mixed(name, mixing=True)
    for name in ASSEMBLE_MIXED_FOOD:
        out[name] = draw_mixed(name)

    out['Cup'] = draw_cup()
    out['CupTile'] = draw_tile(draw_cup())
    for name in ASSEMBLE_MIXED_CUP_FOOD + ASSEMBLE_MIXING_CUP_FOOD:
        out[name] = draw_cup_with(name)

    out['blender'] = draw_blender()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                    help='既にあるファイルも上書きする(本番の絵を差し替え済みなら注意)')
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    GRAPHICS.mkdir(parents=True, exist_ok=True)

    made, skipped = 0, 0
    for name, surface in build().items():
        path = GRAPHICS / f'{name}.png'
        if path.exists() and not args.force:
            skipped += 1
            continue
        pygame.image.save(surface, str(path))
        made += 1

    print(f'生成 {made} 件 / 既存につきスキップ {skipped} 件 -> {GRAPHICS}')


if __name__ == '__main__':
    main()
