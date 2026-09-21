"""キャラの絵を、向き(前・後ろ・横)ごとに作る。

「手を出す先」が向きで決まるので、どちらを向いているかが絵で分かる
ようにする。1マスは画面では 40px しかないので、形は思いきり簡単に
する(丸い顔・コック帽・目・体)。大きさは元の絵と同じ 300x300。

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

S = 300                       # 元の絵と同じ大きさ
LINE = (22, 22, 22, 255)      # 輪郭
WHITE = (254, 254, 254, 255)
HAIR = (106, 63, 0, 255)
EYE = (20, 20, 20, 255)
W = 10                        # 線の太さ(小さく描いても見えるように太め)

# 色ごとの (肌, 服)
COLORS = {
    'blue': ((244, 203, 137, 255), (127, 178, 240, 255)),
    'magenta': ((255, 221, 181, 255), (255, 0, 226, 255)),
}


def base(color):
    """帽子・顔・体だけの土台。向きによらず共通。"""
    skin, body = COLORS[color]
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 体(肩)
    d.rounded_rectangle([86, 228, 214, 292], radius=30,
                        fill=body, outline=LINE, width=W)
    # 顔
    d.ellipse([74, 92, 226, 244], fill=skin, outline=LINE, width=W)
    # コック帽(ふくらみ + つば)
    d.rounded_rectangle([84, 24, 216, 104], radius=40,
                        fill=WHITE, outline=LINE, width=W)
    d.rounded_rectangle([76, 88, 224, 124], radius=16,
                        fill=WHITE, outline=LINE, width=W)
    return img, d


def face_front(color):
    img, d = base(color)
    for x in (126, 174):
        d.ellipse([x - 13, 158, x + 13, 190], fill=EYE)
    d.arc([132, 186, 168, 214], 20, 160, fill=LINE, width=W)
    return img


def face_back(color):
    img, d = base(color)
    # 後頭部は髪でおおう
    d.ellipse([74, 92, 226, 244], fill=HAIR, outline=LINE, width=W)
    d.rounded_rectangle([84, 24, 216, 104], radius=40,
                        fill=WHITE, outline=LINE, width=W)
    d.rounded_rectangle([76, 88, 224, 124], radius=16,
                        fill=WHITE, outline=LINE, width=W)
    return img


def face_right(color):
    skin, body = COLORS[color]
    img, d = base(color)
    head = [74, 92, 226, 244]
    # 鼻(右の出っぱり)を先に描く
    d.pieslice([196, 150, 256, 210], 270, 90, fill=skin, outline=LINE, width=W)
    d.ellipse(head, fill=HAIR)                       # まず全部を髪に
    # 顔は右へずらして描く。はみ出した左側が後ろ髪(三日月)になる。
    d.ellipse([74 + 34, 92 + 8, 226 + 6, 244 - 6], fill=skin)
    d.ellipse(head, outline=LINE, width=W)           # 顔の輪郭を引き直す
    # 帽子を描き直す(顔の上に乗せる)
    d.rounded_rectangle([84, 24, 216, 104], radius=40,
                        fill=WHITE, outline=LINE, width=W)
    d.rounded_rectangle([76, 88, 224, 124], radius=16,
                        fill=WHITE, outline=LINE, width=W)
    # 目は右寄りに1つ
    d.ellipse([174, 156, 202, 190], fill=EYE)
    return img


def main():
    for color in COLORS:
        right = face_right(color)
        images = {
            'front': face_front(color),
            'back': face_back(color),
            'right': right,
            'left': right.transpose(Image.FLIP_LEFT_RIGHT),
        }
        for name, im in images.items():
            path = OUT / f'agent-{color}-{name}.png'
            im.save(path)
            print('作りました:', path.name)


if __name__ == '__main__':
    main()
