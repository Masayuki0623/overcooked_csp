"""スペース押下時に出る「指示カード」パネル(pygame 描画)。

画面構成:
    左50% = ゲーム画面本体(押下時点の見た目をそのまま表示)
    右50% = 指示UIパネル

右パネルの縦方向の並び:
    1. タイマー(右上, 5秒のラジアルワイプ, 緑→黄→赤)
    2. 見出し「つぎの指示をえらんでください」
    3. 環境マップ(簡易表示の帯)
    4. 指示カード一覧(縦リスト)

タイマーは見た目上の演出であり、0 になっても自動選択や強制終了はしない
(0 のまま表示し続け、選択されるまで待つ)。
"""
import pygame

from gym_cooking.misc.game.game import get_image

PANEL_BG = (247, 245, 240)
CARD_BG = (255, 255, 255)
CARD_BG_HOVER = (232, 242, 254)
CARD_BORDER = (214, 210, 202)
CARD_BORDER_HOVER = (66, 133, 244)
TEXT_MAIN = (34, 34, 34)
TEXT_SUB = (122, 118, 112)
ICON_BG = (243, 240, 234)

COUNTDOWN_SECONDS = 5.0
CARD_GAP = 8
MIN_CARD_W = 104
MIN_CARD_H = 44
MAX_CARD_H = 84
TIMER_RADIUS = 30
TIMER_THICKNESS = 9

# 緑 -> 黄 -> 赤 の3色補間
TIMER_COLOR_STOPS = ((76, 175, 80), (255, 193, 7), (229, 57, 53))

# 動詞ごとにカードの色を変える(切る=緑 / 調理=橙 / 提供=青)
VERB_STYLE = {
    'chop':  {'bg': (233, 246, 234), 'hover': (211, 240, 214),
              'border': (102, 187, 106), 'text': (27, 94, 32)},
    'cook':  {'bg': (255, 244, 226), 'hover': (255, 232, 196),
              'border': (255, 167, 38), 'text': (191, 87, 0)},
    'serve': {'bg': (228, 240, 253), 'hover': (205, 229, 252),
              'border': (66, 165, 245), 'text': (13, 71, 161)},
}

INGREDIENT_JP = {'onion': 'たまねぎ', 'tomato': 'トマト', 'lettuce': 'レタス'}
VERB_ACTION_JP = {'chop': '切って', 'cook': '調理して', 'serve': '提供して'}


def _jp_font(size, bold=False):
    """メイリオ優先で日本語フォントを取得する(無ければ順にフォールバック)。"""
    for name in ('Meiryo', 'メイリオ', 'Yu Gothic UI', 'MS Gothic', 'Noto Sans CJK JP'):
        try:
            path = pygame.font.match_font(name, bold=bold)
        except Exception:
            path = None
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size, bold=bold)


def _lerp_color(c0, c1, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def timer_color(progress):
    """progress: 0.0(開始) -> 1.0(終了) を 緑→黄→赤 で補間する。"""
    progress = max(0.0, min(1.0, progress))
    if progress <= 0.5:
        return _lerp_color(TIMER_COLOR_STOPS[0], TIMER_COLOR_STOPS[1], progress / 0.5)
    return _lerp_color(TIMER_COLOR_STOPS[1], TIMER_COLOR_STOPS[2], (progress - 0.5) / 0.5)


def _ingredients_of(obj):
    base = str(obj).replace(' soup', '')
    return [p.strip() for p in base.split('-') if p.strip()]


def card_label(verb, obj):
    """カードの小さい文字(素材名・料理名)。"""
    ings = _ingredients_of(obj)
    if verb == 'chop':
        return INGREDIENT_JP.get(ings[0] if ings else '', str(obj))
    names = [INGREDIENT_JP.get(i, i) for i in ings]
    if len(names) == 1:
        return f"{names[0]}スープ"
    return "・".join(names) + "スープ"


def card_action(verb):
    return VERB_ACTION_JP.get(verb, str(verb))


def card_icon_name(verb, obj):
    """既存のゲーム内画像(misc/game/graphics/*.png)のファイル名を決める。"""
    ings = [i.capitalize() for i in _ingredients_of(obj)]
    if not ings:
        return None
    if verb == 'chop':
        return f"Fresh{ings[0]}"
    prefix = 'Chopped' if verb == 'cook' else 'Cooked'
    return "-".join(f"{prefix}{i}" for i in ings)


def _load_icon(name, size):
    if not name:
        return None
    try:
        image = get_image(f"misc/game/graphics/{name}.png")
    except Exception:
        return None
    # convert_alpha() は display 未設定だと例外になるので、設定済みのときだけ使う
    if pygame.display.get_surface() is not None:
        try:
            image = image.convert_alpha()
        except pygame.error:
            pass
    try:
        return pygame.transform.smoothscale(image, (size, size))
    except Exception:
        return None


def _draw_circular_icon(surface, icon, center, radius, framed=True):
    if framed:
        pygame.draw.circle(surface, ICON_BG, center, radius)
    if icon is None:
        return
    # 円形にクリップして貼る
    mask = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)
    inner = pygame.transform.smoothscale(icon, (int(radius * 1.6), int(radius * 1.6)))
    holder = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    holder.blit(inner, inner.get_rect(center=(radius, radius)))
    holder.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    surface.blit(holder, (center[0] - radius, center[1] - radius))
    if framed:
        pygame.draw.circle(surface, CARD_BORDER, center, radius, 2)


def _draw_radial_timer(surface, center, remaining, total):
    """残り時間のラジアルワイプ。時計回りに閉じていく。"""
    elapsed = total - remaining
    progress = 0.0 if total <= 0 else max(0.0, min(1.0, elapsed / total))
    color = timer_color(progress)

    pygame.draw.circle(surface, (233, 230, 224), center, TIMER_RADIUS, TIMER_THICKNESS)

    if remaining > 0:
        # 12時方向から時計回りに、残りぶんだけ描く
        box = pygame.Rect(0, 0, TIMER_RADIUS * 2, TIMER_RADIUS * 2)
        box.center = center
        import math
        start = math.pi / 2 - 2 * math.pi * (1.0 - progress)
        pygame.draw.arc(surface, color, box, start, math.pi / 2, TIMER_THICKNESS)

    font = _jp_font(26, bold=True)
    # 0 になったら 0 のまま表示し続ける(自動選択はしない)
    text = font.render(str(int(max(0, round(remaining + 0.4999)))), True, color)
    surface.blit(text, text.get_rect(center=center))


class InstructionPanel:
    """指示カード画面。選ばれた候補(display, payload)、またはキャンセル時 None を返す。"""

    def __init__(self, candidates, env_summary=None):
        self.candidates = candidates or []
        self.env_summary = env_summary or {}
        self.card_rects = []

    def _draw_env_strip(self, surface, rect):
        pygame.draw.rect(surface, (238, 244, 238), rect, border_radius=8)
        pygame.draw.rect(surface, (219, 228, 219), rect, 1, border_radius=8)

        title_font = _jp_font(13, bold=True)
        body_font = _jp_font(11)

        # 絵文字はメイリオに字形が無く豆腐(□)になるため、印は自前で描く
        pin = (rect.x + 14, rect.y + 14)
        pygame.draw.circle(surface, (76, 140, 90), pin, 4)
        pygame.draw.circle(surface, (240, 248, 240), pin, 2)

        area = self.env_summary.get('area', 'キッチン')
        detail = self.env_summary.get('detail', '')
        surface.blit(title_font.render(area, True, TEXT_MAIN), (rect.x + 26, rect.y + 6))
        if detail:
            surface.blit(body_font.render(detail, True, TEXT_SUB),
                         (rect.x + 12, rect.y + 25))

    def plan_layout(self, view_w, view_h):
        """候補が全部収まる列数とカードサイズを事前に計算する。

        1〜3列を試し、すべてのカードが view に収まる中で一番大きくなる構成を選ぶ。
        列が増えるとカードは横に細くなるので、幅と高さの小さい方(=見た目の窮屈さ)
        が最大になる構成を採用する。
        """
        n = len(self.candidates)
        if n == 0:
            return 1, 0, 0
        best = None
        for cols in (1, 2, 3):
            if cols > n:
                break
            rows = (n + cols - 1) // cols
            card_w = (view_w - CARD_GAP * (cols - 1)) // cols
            card_h = (view_h - CARD_GAP * (rows - 1)) // rows
            if card_w < MIN_CARD_W or card_h < MIN_CARD_H:
                continue
            card_h = min(card_h, MAX_CARD_H)
            score = min(card_w, card_h * 2)
            if best is None or score > best[0]:
                best = (score, cols, card_w, card_h)
        if best is None:
            # どう並べても最小サイズに満たないときは、最小値より小さくなっても
            # view からはみ出さないことを優先する(はみ出す方が実害が大きい)。
            cols = 3 if n > 6 else 2
            rows = (n + cols - 1) // cols
            card_w = (view_w - CARD_GAP * (cols - 1)) // cols
            card_h = max(24, (view_h - CARD_GAP * (rows - 1)) // rows)
            return cols, card_w, card_h
        return best[1], best[2], best[3]

    @staticmethod
    def _shrink_to_fit(text, max_width, max_size, min_size, color, bold=False):
        """収まる範囲で一番大きいフォントを選ぶ。それでも無理なら末尾を省略する。"""
        for size in range(max_size, min_size - 1, -1):
            font = _jp_font(size, bold=bold)
            if font.size(text)[0] <= max_width:
                return font.render(text, True, color)
        return InstructionPanel._fit_text(_jp_font(min_size, bold=bold), text, max_width, color)

    @staticmethod
    def _fit_text(font, text, max_width, color):
        """max_width に収まるように末尾を省略して描画用 Surface を返す。"""
        if font.size(text)[0] <= max_width:
            return font.render(text, True, color)
        ellipsis = "…"
        trimmed = text
        while trimmed and font.size(trimmed + ellipsis)[0] > max_width:
            trimmed = trimmed[:-1]
        return font.render((trimmed + ellipsis) if trimmed else ellipsis, True, color)

    def _draw_card(self, surface, rect, verb, obj, hovered):
        """イラストを1文字として扱い「[たまねぎの絵]を切って」と読ませる。

        イラストの上には、ふりがなの要領で素材名を小さく添える。
        カードの地色と文字色は動詞ごとに変える。
        """
        style = VERB_STYLE.get(verb, VERB_STYLE['chop'])
        pygame.draw.rect(surface, style['hover'] if hovered else style['bg'],
                         rect, border_radius=10)
        pygame.draw.rect(surface, style['border'], rect, 2 if hovered else 1, border_radius=10)

        tail = f"を{card_action(verb)}"
        ruby = card_label(verb, obj)
        inner_w = rect.width - 12

        # 「イラスト + 文字」が1行に収まるまで、文字とイラストを一緒に縮める
        body_size = max(8, min(rect.height // 5, 13))
        while True:
            body_font = _jp_font(body_size, bold=True)
            ruby_font = _jp_font(max(6, body_size - 3))
            icon_d = max(14, int(body_size * 2.0))
            tail_w = body_font.size(tail)[0]
            if icon_d + 3 + tail_w <= inner_w or body_size <= 8:
                break
            body_size -= 1

        tail_surf = body_font.render(tail, True, style['text'])
        # ふりがなはその行に他の要素が無いので、カード幅いっぱいまで使ってよい
        ruby_surf = self._fit_text(ruby_font, ruby, inner_w, style['text'])

        # ふりがなの高さぶんだけ本文を下げて、イラスト上に重ならないようにする
        ruby_h = ruby_surf.get_height()
        content_h = ruby_h + max(icon_d, tail_surf.get_height())
        top = rect.centery - content_h // 2
        line_cy = top + ruby_h + max(icon_d, tail_surf.get_height()) // 2

        total_w = icon_d + 3 + tail_surf.get_width()
        x = rect.centerx - total_w // 2

        icon_cx = x + icon_d // 2
        _draw_circular_icon(surface, _load_icon(card_icon_name(verb, obj), icon_d),
                            (icon_cx, line_cy), icon_d // 2, framed=False)
        # ふりがなはイラストの真上に置くが、カードからはみ出さないように寄せる
        ruby_rect = ruby_surf.get_rect(centerx=icon_cx, top=top)
        ruby_rect.left = max(ruby_rect.left, rect.left + 6)
        ruby_rect.right = min(ruby_rect.right, rect.right - 6)
        surface.blit(ruby_surf, ruby_rect)
        surface.blit(tail_surf, tail_surf.get_rect(left=x + icon_d + 3, centery=line_cy))

    def _draw_cards(self, surface, top, panel_rect, mouse_pos):
        self.card_rects = []
        if not self.candidates:
            return

        view_x = panel_rect.x + 14
        view_w = panel_rect.width - 28
        view_h = panel_rect.bottom - 24 - top
        cols, card_w, card_h = self.plan_layout(view_w, view_h)

        for i, (display, payload) in enumerate(self.candidates):
            row, col = divmod(i, cols)
            rect = pygame.Rect(view_x + col * (card_w + CARD_GAP),
                               top + row * (card_h + CARD_GAP), card_w, card_h)
            verb = payload.get('verb') if isinstance(payload, dict) else None
            obj = payload.get('obj') if isinstance(payload, dict) else None
            if verb is None or obj is None:
                verb, obj = 'chop', str(display)
            self._draw_card(surface, rect, verb, obj, rect.collidepoint(mouse_pos))
            self.card_rects.append((rect, (display, payload)))

    def render(self, surface):
        """パネル1枚分を surface に描く(ゲーム画面には触れない)。"""
        rect = surface.get_rect()
        surface.fill(PANEL_BG)

        remaining = max(0.0, COUNTDOWN_SECONDS - (pygame.time.get_ticks() / 1000.0 - self._started))
        _draw_radial_timer(surface, (rect.right - 16 - TIMER_RADIUS, 16 + TIMER_RADIUS),
                           remaining, COUNTDOWN_SECONDS)

        heading_font = _jp_font(17, bold=True)
        surface.blit(heading_font.render("つぎの指示を", True, TEXT_MAIN), (16, 22))
        surface.blit(heading_font.render("えらんでください", True, TEXT_MAIN), (16, 44))

        strip = pygame.Rect(16, 16 + TIMER_RADIUS * 2 + 12, rect.width - 32, 46)
        self._draw_env_strip(surface, strip)
        self._draw_cards(surface, strip.bottom + 14, rect, self._mouse)

        hint_font = _jp_font(10)
        surface.blit(hint_font.render("クリックで選択 / Esc でキャンセル", True, TEXT_SUB),
                     (16, rect.bottom - 20))

    def run_windowed(self, size, position=None):
        """ゲーム窓とは別のウィンドウを開いて選択を待つ。

        ゲーム側の画面には一切触れないので、描画の奪い合い(点滅)も、
        描画途中のコピーによるオブジェクト欠けも起きない。
        """
        from pygame._sdl2.video import Window, Renderer, Texture

        window = Window("つぎの指示", size=size, position=position)
        renderer = Renderer(window)
        canvas = pygame.Surface(size)
        clock = pygame.time.Clock()
        self._started = pygame.time.get_ticks() / 1000.0
        self._mouse = (-1, -1)
        try:
            window.focus()
        except Exception:
            pass

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return None
                    if event.type == pygame.WINDOWCLOSE and self._is_panel_event(event, window):
                        return None
                    if event.type == pygame.MOUSEMOTION and self._is_panel_event(event, window):
                        self._mouse = event.pos
                    if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                            and self._is_panel_event(event, window)):
                        for rect, choice in self.card_rects:
                            if rect.collidepoint(event.pos):
                                return choice

                self.render(canvas)
                texture = Texture.from_surface(renderer, canvas)
                renderer.clear()
                texture.draw()
                renderer.present()
                clock.tick(30)
        finally:
            try:
                window.destroy()
            except Exception:
                pass

    @staticmethod
    def _is_panel_event(event, window):
        """このパネルのウィンドウで起きたイベントか(判別できなければ受け入れる)。"""
        ev_window = getattr(event, 'window', None)
        if ev_window is None:
            return True
        try:
            return ev_window.id == window.id
        except Exception:
            return True

