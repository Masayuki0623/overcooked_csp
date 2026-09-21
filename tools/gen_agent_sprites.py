"""キャラの絵を、向き(前・後ろ・横)ごとに作る。

「手を出す先」が向きで決まるようになったので、どちらを向いているかが
絵で分かるようにする。元の絵(agent-blue.png / agent-magenta.png)と
同じ雰囲気・同じ大きさ(300x300)で、次の4枚を作る。

    agent-<color>-front.png   手前(下)を向いている
    agent-<color>-back.png    奥(上)を向いている
    agent-<color>-right.png   右を向いている
    agent-<color>-left.png    左を向いている(右向きの左右反転)

    python tools/gen_agent_sprites.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'testbed-cooking' / 'gym_cooking' / 'misc' / 'game' / 'graphics'

S = 300                      # 元の絵と同じ大きさ
LINE = (22, 22, 22, 255)     # 輪郭
WHITE = (254, 254, 254, 255)
SHADE = (222, 222, 222, 255)  # 帽子の影
HAIR = (106, 63, 0, 255)
CHEEK = (252, 165, 145, 255)

# 色ごとの (肌, 服, 服の影)
COLORS = {
    'blue': ((244, 203, 137, 255), (127, 178, 240, 255), (63, 115, 208, 255)),
    'magenta': ((255, 221, 181, 255), (255, 0, 226, 255), (178, 0, 158, 255)),
}

W = 8    # 線の太さ


def new_canvas():
    return Image.new('RGBA', (S, S), (0, 0, 0, 0))


def draw_hat(d, cx=150, cy=78, flip=0):
    """コック帽。cx を動かすと横顔用に寄せられる。"""
    # ふくらみ3つ
    for dx, dy, r in ((-58, 6, 46), (0, -14, 52), (58, 6, 46)):
        d.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r],
                  fill=WHITE, outline=LINE, width=W)
    # 帽子の縁(かぶり口)
    d.rounded_rectangle([cx - 68, cy + 34, cx + 68, cy + 74], radius=14,
                        fill=WHITE, outline=LINE, width=W)
    # 影(左側)
    d.pieslice([cx - 66, cy + 36, cx - 18, cy + 72], 90, 270, fill=SHADE)


def draw_body(d, skin, body, shadow, back=False):
    """肩と襟。back=True なら襟を描かず背中に見せる。"""
    d.rounded_rectangle([88, 236, 212, 292], radius=26,
                        fill=body, outline=LINE, width=W)
    if back:
        d.line([(150, 240), (150, 288)], fill=shadow, width=6)
    else:
        d.pieslice([88, 214, 212, 286], 200, 340, fill=shadow)


def face_front(color):
    skin, body, shadow = COLORS[color]
    img = new_canvas()
    d = ImageDraw.Draw(img)
    draw_body(d, skin, body, shadow)
    # 髪(顔の後ろ)
    d.ellipse([56, 104, 244, 258], fill=HAIR, outline=LINE, width=W)
    # 顔
    d.ellipse([70, 112, 230, 248], fill=skin, outline=LINE, width=W)
    draw_hat(d)
    # 目・ほお・口
    for x in (118, 182):
        d.ellipse([x - 12, 156, x + 12, 186], fill=(20, 20, 20, 255))
    d.ellipse([80, 176, 114, 204], fill=CHEEK)
    d.ellipse([186, 176, 220, 204], fill=CHEEK)
    d.arc([126, 186, 174, 222], 20, 160, fill=LINE, width=W)
    return img


def face_back(color):
    skin, body, shadow = COLORS[color]
    img = new_canvas()
    d = ImageDraw.Draw(img)
    draw_body(d, skin, body, shadow, back=True)
    # 後頭部は髪だけ。下側を少し伸ばして襟足にする。
    d.ellipse([66, 108, 234, 252], fill=HAIR, outline=LINE, width=W)
    d.rounded_rectangle([96, 200, 204, 250], radius=24, fill=HAIR,
                        outline=LINE, width=W)
    d.ellipse([72, 114, 228, 246], fill=HAIR)     # 内側の線を消す
    draw_hat(d)
    return img


def face_right(color):
    skin, body, shadow = COLORS[color]
    img = new_canvas()
    d = ImageDraw.Draw(img)
    draw_body(d, skin, body, shadow)
    # 後ろ髪(左側)
    d.ellipse([52, 108, 210, 252], fill=HAIR, outline=LINE, width=W)
    # 横顔(右に寄せる)
    d.ellipse([84, 112, 236, 246], fill=skin, outline=LINE, width=W)
    # 髪を前髪ぎみに少し重ねる
    d.pieslice([84, 104, 236, 210], 175, 300, fill=HAIR, outline=LINE, width=W)
    # 鼻(右のふくらみ)
    d.pieslice([214, 158, 254, 198], 270, 90, fill=skin, outline=LINE, width=W)
    draw_hat(d, cx=158)
    # 目(1つ)・ほお・口
    d.ellipse([186, 160, 210, 190], fill=(20, 20, 20, 255))
    d.ellipse([162, 186, 194, 212], fill=CHEEK)
    d.arc([192, 196, 226, 220], 300, 40, fill=LINE, width=W)
    return img


def main():
    for color in COLORS:
        front = face_front(color)
        back = face_back(color)
        right = face_right(color)
        left = right.transpose(Image.FLIP_LEFT_RIGHT)
        for name, im in (('front', front), ('back', back),
                         ('right', right), ('left', left)):
            path = OUT / f'agent-{color}-{name}.png'
            im.save(path)
            print('作りました:', path.name)


if __name__ == '__main__':
    main()
