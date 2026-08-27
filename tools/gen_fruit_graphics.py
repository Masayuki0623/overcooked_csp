# -*- coding: utf-8 -*-
"""フルーツ/ジュース工程の画像を生成する。

既存の野菜アート(FreshTomato.png など)はフラットなベクター調のクリップアートで、
輪郭線が無く、ベタ塗りの base 色と一段暗い shade 色で立体感を出している。
ゲーム側は全部 40px タイルまで縮小して描くので、細かい模様は入れない。

使い方:
    python tools/gen_fruit_graphics.py              # 全部書き出す
    python tools/gen_fruit_graphics.py --sheet x.png  # 確認用の一覧画像も作る
"""
import argparse
import itertools
import math
import os

from PIL import Image, ImageChops, ImageDraw

SS = 1536          # 描画用キャンバス(この上で描いて縮小する)
OUT = 512          # 書き出しサイズ
HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHICS = os.path.join(HERE, '..', 'testbed-cooking', 'gym_cooking',
                        'misc', 'game', 'graphics')

FRUITS = ['Apple', 'Banana', 'Orange']

# 既存アートの実測色: トマト #FF2D00 / レタス #98C12A / 玉ねぎ #C576BF。
# りんごを普通の赤にするとトマトと見分けが付かないので、暗いクリムゾンにする。
COLORS = {
    'Apple':  dict(base=(198, 40, 40), shade=(142, 27, 27)),
    'Banana': dict(base=(255, 210, 74), shade=(224, 168, 0)),
    'Orange': dict(base=(255, 138, 0), shade=(217, 106, 0)),
}
JUICE = {
    'Apple':  dict(base=(229, 184, 75), shade=(186, 138, 40)),
    'Banana': dict(base=(247, 231, 176), shade=(206, 186, 124)),
    'Orange': dict(base=(255, 158, 27), shade=(211, 118, 0)),
}
FLESH = (253, 243, 217)
FLESH_SHADE = (232, 217, 174)
STEM = (123, 75, 42)
LEAF = (108, 143, 0)
GLASS = (255, 255, 255)
GLASS_SHADE = (226, 226, 226)
GLASS_LINE = (150, 150, 150)
METAL = (110, 110, 110)
METAL_DARK = (78, 78, 78)
METAL_LIGHT = (208, 206, 206)
BUTTON = (244, 126, 0)


# ---------------------------------------------------------------- primitives
def blank():
    return Image.new('RGBA', (SS, SS), (0, 0, 0, 0))


def new_mask():
    return Image.new('L', (SS, SS), 0)


def P(pts):
    return [(x * SS, y * SS) for x, y in pts]


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=96):
    a = math.radians(rot)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * math.cos(a) - y * math.sin(a),
                    cy + x * math.sin(a) + y * math.cos(a)))
    return out


def crescent_pts(cx, cy, R, r, a0, a1, n=48):
    """y下向き座標での三日月。a0..a1 は度。"""
    outer = [(cx + R * math.cos(math.radians(t)), cy + R * math.sin(math.radians(t)))
             for t in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]
    inner = [(cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t)))
             for t in [a1 + (a0 - a1) * i / n for i in range(n + 1)]]
    return outer + inner


def shift(mask, dx, dy):
    return ImageChops.offset(mask, int(dx * SS), int(dy * SS))


def fill(mask, base, shade, off=(-0.035, -0.045)):
    """マスクを base で塗り、offset 分だけずらした側に shade の縁を残す。"""
    layer = blank()
    layer.paste(tuple(shade) + (255,), (0, 0, SS, SS), mask)
    inner = ImageChops.multiply(mask, shift(mask, off[0], off[1]))
    layer.paste(tuple(base) + (255,), (0, 0, SS, SS), inner)
    return layer


def fill_paint(mask, paint, off=(-0.035, -0.045)):
    """paint(RGBA画像)でマスクを塗り、縁だけ暗くする。組み合わせジュース用。"""
    dark = paint.convert('RGB').point(lambda v: int(v * 0.74))
    dark = dark.convert('RGBA')
    dark.putalpha(paint.getchannel('A'))
    layer = blank()
    layer.paste(dark, (0, 0), mask)
    inner = ImageChops.multiply(mask, shift(mask, off[0], off[1]))
    layer.paste(paint, (0, 0), inner)
    return layer


def poly_mask(*polys):
    m = new_mask()
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon(P(p), fill=255)
    return m


def circle_mask(cx, cy, r):
    return poly_mask(ellipse_pts(cx, cy, r, r))


def place(dst, src, cx, cy, scale, angle=0.0):
    im = src
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=False)
    w = max(1, int(SS * scale))
    im = im.resize((w, w), Image.LANCZOS)
    dst.alpha_composite(im, (int(cx * SS - w / 2), int(cy * SS - w / 2)))
    return dst


def over(*layers):
    out = blank()
    for l in layers:
        out.alpha_composite(l)
    return out


def save(layer, name):
    im = layer.resize((OUT, OUT), Image.LANCZOS)
    im.save(os.path.join(GRAPHICS, name + '.png'))
    return name


# ------------------------------------------------------------- whole fruits
def fresh_apple():
    body = poly_mask(ellipse_pts(0.385, 0.575, 0.275, 0.275),
                     ellipse_pts(0.615, 0.575, 0.275, 0.275),
                     ellipse_pts(0.5, 0.63, 0.30, 0.275))
    stem = blank()
    d = ImageDraw.Draw(stem)
    d.line(P([(0.50, 0.40), (0.535, 0.17)]), fill=STEM + (255,),
           width=int(0.045 * SS), joint='curve')
    d.polygon(P(ellipse_pts(0.635, 0.235, 0.135, 0.068, rot=-22)), fill=LEAF + (255,))
    return over(stem, fill(body, **COLORS['Apple']))


def fresh_orange():
    leaf = blank()
    d = ImageDraw.Draw(leaf)
    d.line(P([(0.50, 0.34), (0.515, 0.235)]), fill=STEM + (255,), width=int(0.038 * SS))
    d.polygon(P(ellipse_pts(0.625, 0.245, 0.135, 0.070, rot=-20)), fill=LEAF + (255,))
    body = circle_mask(0.5, 0.585, 0.325)
    return over(leaf, fill(body, **COLORS['Orange']))


def fresh_banana():
    body = poly_mask(crescent_pts(0.5, 0.055, 0.535, 0.365, 40, 140))
    layer = fill(body, off=(-0.03, -0.05), **COLORS['Banana'])
    tips = blank()
    d = ImageDraw.Draw(tips)
    for ang in (40, 140):
        cx = 0.5 + 0.45 * math.cos(math.radians(ang))
        cy = 0.055 + 0.45 * math.sin(math.radians(ang))
        d.polygon(P(ellipse_pts(cx, cy, 0.055, 0.055)), fill=STEM + (255,))
    tips.putalpha(ImageChops.multiply(tips.getchannel('A'), body))
    return over(layer, tips)


WHOLE = {'Apple': fresh_apple, 'Banana': fresh_banana, 'Orange': fresh_orange}


# -------------------------------------------------------------- cut pieces
def piece_apple():
    peel = poly_mask(crescent_pts(0.5, 0.30, 0.345, 0.145, 22, 158))
    flesh = poly_mask(crescent_pts(0.5, 0.30, 0.290, 0.145, 26, 154))
    return over(fill(peel, **COLORS['Apple']),
                fill(flesh, FLESH, FLESH_SHADE, off=(-0.02, -0.03)))


def piece_orange():
    rim = circle_mask(0.5, 0.5, 0.325)
    inner = circle_mask(0.5, 0.5, 0.255)
    layer = over(fill(rim, **COLORS['Orange']),
                 fill(inner, (255, 178, 71), (232, 150, 44), off=(-0.02, -0.03)))
    spokes = blank()
    d = ImageDraw.Draw(spokes)
    for k in range(6):
        t = math.radians(30 + 60 * k)
        d.line(P([(0.5, 0.5), (0.5 + 0.26 * math.cos(t), 0.5 + 0.26 * math.sin(t))]),
               fill=FLESH + (255,), width=int(0.022 * SS))
    spokes.putalpha(ImageChops.multiply(spokes.getchannel('A'), inner))
    return over(layer, spokes)


def piece_banana():
    rim = circle_mask(0.5, 0.5, 0.315)
    inner = circle_mask(0.5, 0.5, 0.245)
    core = circle_mask(0.5, 0.5, 0.075)
    return over(fill(rim, **COLORS['Banana']),
                fill(inner, FLESH, FLESH_SHADE, off=(-0.02, -0.03)),
                fill(core, (240, 224, 176), (222, 204, 150), off=(-0.01, -0.02)))


PIECE = {'Apple': piece_apple, 'Banana': piece_banana, 'Orange': piece_orange}

# 切った食材の並べ方。1種類なら3切れ、2種類以上なら2切れずつ。
LAYOUT = {
    3: [(0.34, 0.60, 12, 0.60), (0.58, 0.44, -14, 0.60), (0.63, 0.70, 20, 0.60)],
    4: [(0.33, 0.44, 14, 0.52), (0.62, 0.37, -16, 0.52),
        (0.38, 0.70, -8, 0.52), (0.67, 0.64, 22, 0.52)],
    6: [(0.30, 0.40, 12, 0.44), (0.55, 0.31, -14, 0.44), (0.73, 0.45, 20, 0.44),
        (0.29, 0.65, -10, 0.44), (0.51, 0.58, 16, 0.44), (0.68, 0.72, -20, 0.44)],
}


def chopped(fruits):
    pieces = []
    n = 3 if len(fruits) == 1 else 2
    for f in fruits:
        pieces += [f] * n
    slots = LAYOUT[len(pieces)]
    # 上(y が小さい)から描くと重なりが自然になる
    order = sorted(range(len(pieces)), key=lambda i: slots[i][1])
    canvas = blank()
    cache = {f: PIECE[f]() for f in set(fruits)}
    for i in order:
        cx, cy, rot, sc = slots[i]
        place(canvas, cache[pieces[i]], cx, cy, sc, rot)
    return canvas


# ------------------------------------------------------------- chopping bar
def chopping(fruit, stage):
    """丸ごとの絵を縦に切り分けて、切断の進み具合を出す。"""
    whole = WHOLE[fruit]()
    cuts = stage                      # 1..3 本の切れ目
    xs = [0.0] + [0.5 + (i - (cuts - 1) / 2.0) * 0.16 for i in range(cuts)] + [1.0]
    canvas = blank()
    for i in range(len(xs) - 1):
        left, right = int(xs[i] * SS), int(xs[i + 1] * SS)
        band = whole.crop((left, 0, right, SS))
        if i > 0:
            # 切り口をクリーム色で見せる
            face = Image.new('RGBA', band.size, FLESH + (255,))
            fm = Image.new('L', band.size, 0)
            ImageDraw.Draw(fm).rectangle(
                [0, 0, max(1, int(0.045 * SS)), SS], fill=255)
            band.paste(face, (0, 0), ImageChops.multiply(fm, band.getchannel('A')))
        dx = int((i - (len(xs) - 2) / 2.0) * 0.030 * SS * stage / 3.0)
        canvas.alpha_composite(band, (max(0, left + dx), 0))
    return canvas


# -------------------------------------------------------------------- juice
# 既存の CookedTomato.png は「鍋の中身を丸い器として真上から見た絵」なので、
# ミキサーの中身もそれに合わせて丸くする。四角い塊にするとパンに見えてしまう。
POOL_C, POOL_R, POOL_IN = (0.5, 0.545), 0.355, 0.295


def darken(img, k):
    d = img.convert('RGB').point(lambda v: int(v * k)).convert('RGBA')
    d.putalpha(img.getchannel('A'))
    return d


def bands(fruits, x0, x1, tilt=0.05):
    """果物の数だけ縦の帯に塗り分けた画像。境目は少し斜めにする。"""
    img = Image.new('RGBA', (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    n = len(fruits)
    for i, f in enumerate(fruits):
        a = x0 + (x1 - x0) * i / float(n)
        b = x0 + (x1 - x0) * (i + 1) / float(n)
        left = -0.2 if i == 0 else a
        right = 1.2 if i == n - 1 else b
        d.polygon(P([(left - tilt, -0.2), (right - tilt, -0.2),
                     (right + tilt, 1.2), (left + tilt, 1.2)]),
                  fill=tuple(JUICE[f]['base']) + (255,))
    return img


def chunk_img():
    m = new_mask()
    ImageDraw.Draw(m).rounded_rectangle(
        [0.30 * SS, 0.38 * SS, 0.70 * SS, 0.62 * SS], radius=0.09 * SS, fill=255)
    return fill(m, FLESH, FLESH_SHADE, off=(-0.02, -0.03))


def juice(fruits, mixing=False):
    cx, cy = POOL_C
    outer = circle_mask(cx, cy, POOL_R)
    inner = circle_mask(cx, cy, POOL_IN)
    paint = bands(fruits, cx - POOL_R, cx + POOL_R)
    layer = blank()
    layer.paste(darken(paint, 0.62), (0, 0), outer)          # 器の縁
    layer = over(layer, fill_paint(inner, paint, off=(-0.025, -0.035)))
    hl = blank()
    ImageDraw.Draw(hl).polygon(
        P(ellipse_pts(cx - 0.115, cy - 0.115, 0.105, 0.042, rot=-32)),
        fill=(255, 255, 255, 130))
    hl.putalpha(ImageChops.multiply(hl.getchannel('A'), inner))
    layer = over(layer, hl)
    if mixing:
        chunks = blank()
        c = chunk_img()
        for px, py, rot in [(0.38, 0.50, 14), (0.56, 0.62, -18), (0.62, 0.45, 8)]:
            place(chunks, c, px, py, 0.20, rot)
        chunks.putalpha(ImageChops.multiply(chunks.getchannel('A'), inner))
        layer = over(layer, chunks)
    return layer


# ---------------------------------------------------------------------- cup
CUP_TOP_Y, CUP_BOT_Y = 0.245, 0.815
CUP_TOP_HW, CUP_BOT_HW = 0.205, 0.140
CUP_R = 0.055


def cup_half_width(y):
    t = (y - CUP_TOP_Y) / (CUP_BOT_Y - CUP_TOP_Y)
    return CUP_TOP_HW + (CUP_BOT_HW - CUP_TOP_HW) * t


def cup_body_pts(inset=0.0):
    """上が広い台形。下の左右の角だけ丸める。"""
    corner_y = CUP_BOT_Y - CUP_R
    hw = cup_half_width(corner_y) - inset
    pts = [(0.5 - CUP_TOP_HW + inset, CUP_TOP_Y),
           (0.5 + CUP_TOP_HW - inset, CUP_TOP_Y),
           (0.5 + hw, corner_y)]
    for k in range(1, 7):                       # 右下の角
        t = math.radians(90.0 * k / 6)
        pts.append((0.5 + hw - CUP_R + CUP_R * math.cos(t),
                    corner_y + CUP_R * math.sin(t)))
    for k in range(6, -1, -1):                  # 左下の角
        t = math.radians(90.0 * k / 6)
        pts.append((0.5 - hw + CUP_R - CUP_R * math.cos(t),
                    corner_y + CUP_R * math.sin(t)))
    pts.append((0.5 - hw, corner_y))
    return pts


def cup_layer(contents=None, mixing=False):
    body = poly_mask(cup_body_pts())
    inner = poly_mask(cup_body_pts(inset=0.022))
    layer = fill(body, GLASS, GLASS_SHADE, off=(-0.030, -0.020))
    if contents:
        surface_y = 0.415
        liquid = new_mask()
        ImageDraw.Draw(liquid).rectangle(
            [0, surface_y * SS, SS, CUP_BOT_Y * SS], fill=255)
        liquid = ImageChops.multiply(liquid, inner)
        paint = bands(contents, 0.5 - CUP_TOP_HW, 0.5 + CUP_TOP_HW, tilt=0.03)
        layer = over(layer, fill_paint(liquid, paint, off=(-0.02, -0.03)))
        top = blank()
        hw = cup_half_width(surface_y) - 0.022
        ImageDraw.Draw(top).polygon(
            P(ellipse_pts(0.5, surface_y, hw, 0.038)), fill=(255, 255, 255, 90))
        top.putalpha(ImageChops.multiply(top.getchannel('A'), inner))
        layer = over(layer, top)
        if mixing:
            chunks = blank()
            c = chunk_img()
            for cx, cy, rot in [(0.44, 0.52, 12), (0.58, 0.63, -16), (0.50, 0.72, 6)]:
                place(chunks, c, cx, cy, 0.16, rot)
            chunks.putalpha(ImageChops.multiply(chunks.getchannel('A'), liquid))
            layer = over(layer, chunks)
    outline = blank()
    d = ImageDraw.Draw(outline)
    d.line(P(cup_body_pts() + [cup_body_pts()[0]]), fill=GLASS_LINE + (255,),
           width=int(0.020 * SS), joint='curve')
    d.polygon(P(ellipse_pts(0.5, CUP_TOP_Y, CUP_TOP_HW, 0.048)),
              fill=GLASS + (255,), outline=GLASS_LINE + (255,), width=int(0.020 * SS))
    return over(layer, outline)


# ------------------------------------------------------------------ blender
def blender():
    base = new_mask()
    ImageDraw.Draw(base).rounded_rectangle(
        [0.28 * SS, 0.615 * SS, 0.72 * SS, 0.865 * SS], radius=0.055 * SS, fill=255)
    jar = poly_mask([(0.335, 0.235), (0.665, 0.235), (0.625, 0.625), (0.375, 0.625)])
    lid = new_mask()
    ImageDraw.Draw(lid).rounded_rectangle(
        [0.315 * SS, 0.150 * SS, 0.685 * SS, 0.250 * SS], radius=0.035 * SS, fill=255)
    knob = blank()
    ImageDraw.Draw(knob).polygon(P(ellipse_pts(0.60, 0.745, 0.045, 0.045)),
                                 fill=BUTTON + (255,))
    return over(fill(base, METAL, METAL_DARK, off=(-0.03, -0.03)),
                fill(jar, METAL_LIGHT, (170, 168, 168), off=(-0.035, -0.02)),
                fill(lid, (158, 158, 158), (120, 120, 120), off=(-0.02, -0.02)),
                knob)


# -------------------------------------------------------------------- tiles
def tile_of(src, slots):
    canvas = blank()
    for cx, cy, sc in slots:
        place(canvas, src, cx, cy, sc)
    return canvas


# --------------------------------------------------------------------- main
def combos(items):
    out = []
    for r in range(1, len(items) + 1):
        out += [list(c) for c in itertools.combinations(sorted(items), r)]
    return out


def generate():
    made = []

    for f in FRUITS:
        made.append(save(WHOLE[f](), 'Fresh' + f))
        for s in (1, 2, 3):
            made.append(save(chopping(f, s), 'Chopping%d%s' % (s, f)))
        # 切っている途中の食材を手に持つと full_name が番号なしの
        # 'Chopping<果物>' になり、この名前で描画される。
        made.append(save(chopping(f, 2), 'Chopping' + f))

    for combo in combos(FRUITS):
        made.append(save(chopped(combo), '-'.join('Chopped' + f for f in combo)))
        made.append(save(juice(combo, False), '-'.join('Mixed' + f for f in combo)))
        made.append(save(juice(combo, True), '-'.join('Mixing' + f for f in combo)))
        made.append(save(cup_layer(combo, False),
                         '-'.join(['Cup'] + ['Mixed' + f for f in combo])))
        made.append(save(cup_layer(combo, True),
                         '-'.join(['Cup'] + ['Mixing' + f for f in combo])))

    cup = cup_layer()
    made.append(save(cup, 'Cup'))
    made.append(save(blender(), 'blender'))
    made.append(save(tile_of(cup, [(0.30, 0.42, 0.62), (0.70, 0.42, 0.62),
                                   (0.50, 0.60, 0.66)]), 'CupTile'))
    for f in FRUITS:
        made.append(save(tile_of(WHOLE[f](), [(0.33, 0.40, 0.60), (0.67, 0.40, 0.60),
                                              (0.50, 0.63, 0.62)]), 'Fresh%sTile' % f))
    return made


def contact_sheet(path):
    """40px に縮めた見え方を並べた確認用の画像(背景はカウンターの色)。"""
    names = [n[:-4] for n in sorted(os.listdir(GRAPHICS)) if n.endswith('.png')]
    keep = [n for n in names if any(k in n for k in
            ('Apple', 'Banana', 'Orange', 'Cup', 'blender'))]
    keep += ['FreshTomato', 'ChoppedTomato', 'plate', 'pot', 'PlateTile']
    cols, cell = 9, 96
    rows = (len(keep) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * cell), (220, 170, 110))
    for i, n in enumerate(keep):
        im = Image.open(os.path.join(GRAPHICS, n + '.png')).convert('RGBA')
        im = im.resize((40, 40), Image.LANCZOS).resize((80, 80), Image.NEAREST)
        x, y = (i % cols) * cell + 8, (i // cols) * cell + 8
        sheet.paste(im, (x, y), im)
    sheet.save(path)
    return len(keep)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    made = generate()
    print('generated %d images' % len(made))
    if a.sheet:
        print('sheet: %d tiles' % contact_sheet(a.sheet))
