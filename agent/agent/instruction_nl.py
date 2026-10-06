# -*- coding: utf-8 -*-
"""自然言語の指示を、いまの指示の候補(記号)に対応づける。

参加者が書いた文を LLM に読ませ、候補の一覧(動詞・対象・個数の id)の中から
当てはまるものを選ばせる。候補に無いものや曖昧な文は却下させる。LLM は
一覧から選ぶだけなので、存在しない作業が返ってくることはない。

    menu   = menu_of(candidates)                 # [(id, 説明), ...]
    result = interpret(text, menu, orders_ja)    # {'decision', 'tasks', 'reject_reason', 'message', ...}

鍵は環境変数(GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)か、
リポジトリ直下の google_api_key.txt / openai_api_key.txt / anthropic_api_key.txt。
"""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 使うモデル。先頭から順に試し、呼べなければ次へ(鍵が無い・残高が無い・
# 混雑など)。tools/llm_instruction_bench.py で測って決める(2026-10-06、63件):
#   gpt-5.4-mini          : 58/63、応答 中央値 1.1 秒  <- これを使う
#   gpt-5.4-nano          : 50/63、1.2 秒(「作って」を煮るまでと取り違えやすい)
#   gemini-3.5-flash-lite : 61/63、1.3 秒(無料枠。混むと10秒超。使わない)
# 実験では参加者ごとにモデルが変わらないよう、1つだけにしておく。
DEFAULT_MODELS = ['gpt-5.4-mini']
DEFAULT_MODEL = DEFAULT_MODELS[0]
# 1回の指示で AI に任せてよい量の上限。None なら上限なし(仕様: 当てはまる
# 作業はすべて返す。「全部やって」は全部)。数を入れると、超えたら reject。
MAX_DISHES = None       # serve / cook 系(料理まるごと)
MAX_CHOP_UNITS = None   # 切る材料の合計
# 1回の指示で AI がやる工程の数の下限と、全体に占める割合の上限(2026-10-06)。
# 「たまねぎを1つ切って」(工程1つ)は AI の分担が少なすぎるので受けない。
# 候補(鎖)が分かるときだけ数える(候補が無い bench では数えない)。
MIN_AI_STEPS = 2
MAX_AI_SHARE = 0.5
REJECT_REASONS = ['invalid', 'ambiguous', 'too_few', 'too_many', 'error']
# 参加者に見せる却下の文。やわらかく、何をどう書き直せばよいかを伝える
# (「曖昧です」「無効です」は使わない。2026-10-06)
REJECT_JA = {
    'invalid': 'この回の注文にある作業で、AI にやってほしいことを書き直してください。',
    'ambiguous': 'どの作業か、いくつかが決めきれませんでした。料理や材料と個数を入れて、書き直してください。',
    'too_few': 'もう少し多くの作業を頼んでください。切る作業を2つ以上にするか、'
               '料理を煮る・提供するところまで任せる形で書き直してください。',
    'too_many': '一度に頼めるのは全体の半分までです。頼む量を減らして、書き直してください。',
    'error': 'うまく読み取れませんでした。もう一度送ってください。',
}
# LLM が書いた文にこの語があれば、こちらの文に差し替える
HARSH_WORDS = ('曖昧', '無効', '有効では', '不明確', '不適切')


def soft_message(llm_message, reason):
    """LLM の助言が使えればそれ(書き直しの促しを添える)、きつい言い方ならこちらの文。"""
    m = (llm_message or '').strip()
    if not m or any(w in m for w in HARSH_WORDS):
        return REJECT_JA[reason]
    return m if m.endswith('。') else m + '。'

# LLM の出力の形。
#   valid     : 有効な指示か(この回の注文にある作業を、AI に指定しているか)
#   ambiguous : 曖昧か(作業が特定できない / 個数が決まらない)
#   tasks     : 候補の id の一覧(valid かつ ambiguous でないとき)
#   message   : 参加者に見せる短い文
SCHEMA_BASE = {
    'type': 'object',
    'properties': {
        'valid': {'type': 'boolean'},
        'ambiguous': {'type': 'boolean'},
        'tasks': {'type': 'array', 'items': {'type': 'string'}},
        # AI にやらせない工程(「切るのは私がやる」)。ふつうは空
        'exclude_steps': {'type': 'array', 'items': {'type': 'string', 'enum': ['chop', 'cook', 'serve']}},
        'message': {'type': 'string'},
    },
    'required': ['valid', 'ambiguous', 'tasks', 'exclude_steps', 'message'],
}
STEP_JA = {'chop': '切る', 'cook': '煮る', 'serve': '提供する'}


def schema_for(menu_ids):
    s = json.loads(json.dumps(SCHEMA_BASE))
    s['properties']['tasks']['items']['enum'] = list(menu_ids)
    return s


def system_prompt(menu, orders_ja, ingredients_ja='たまねぎ・トマト・レタス'):
    lines = '\n'.join(f'  {i}: {d}' for i, d in menu)
    return f'''あなたは、人と AI が協力して料理を作るゲームの「指示の解釈係」です。
参加者が AI の相方に出した指示の文を読み、下の「できる作業の一覧」から当てはまる作業を選びます。

この回の注文: {orders_ja}
材料は {ingredients_ja} だけです。

できる作業の一覧(id: 内容):
{lines}

出力:
- valid    : 有効な指示なら true。false にするのは次の2つだけ。
             (a) この回の注文に無い料理・材料・個数を指している(例: ジュース、にんじん、注文に無い料理、要る数より多い個数)
             (b) 指示の文ではない(挨拶、質問、感想、自分の分担の宣言だけ。例: こんにちは、私がスープをやります)
             AI に何かをさせようとしている文は、内容が曖昧でも valid は true(例: 「右で作業して」「急いで」「手伝って」「何か切って」は valid=true, ambiguous=true)。
             「全部やって」「全部あなたがやって」は valid=true で、料理をすべて選ぶ。
- ambiguous: 曖昧なら true(valid が true のときだけ見る)。
             (1) 作業が特定できない(場所・方向・速さ・態度・手伝い方だけで、料理や材料が無い。例: 右で作業して、急いで、手伝って、好きにやって、丁寧に、鍋をよろしく、何か切って)
             (2) 切る個数が決まらない(その材料がこの回に2つ要るのに、数が書かれていない。例: 「たまねぎを切って」で、たまねぎが2つ要るとき。「野菜を切って」も同じ)
             料理の種類だけの指定(「スープを作って」「スープを煮て」)は曖昧ではない(その種類の料理をすべて選ぶ)。
- tasks    : valid かつ ambiguous でないとき、当てはまる id をすべて。それ以外は空。
             同じ作業の個数違い(x1 と x2)を両方選んではいけない。個数が決まらなければ ambiguous。
- exclude_steps: 「材料は切らないで」「切るのは私がやる」「提供は私がやる」のように、料理の工程のうち AI に
             やらせないと言われた工程を入れる(chop=切る, cook=煮る, serve=提供)。何も言われなければ空の配列。
             除く工程があっても、料理の指定(serve 系 / cook 系)はそのまま選ぶ(受けるかどうかは後で決める)。
- message  : 参加者に見せる短い日本語。受理なら解釈した内容を1文で、曖昧・無効なら「何を足して書き直せばよいか」を
             やわらかく1文で(「曖昧です」「無効です」「不明確」のような言い方はしない。例: 「たまねぎは2つ要るので、
             切る個数を入れて書き直してください」)。

規則:
- 一覧にある id だけを使う。文に作業が複数あれば、当てはまる id をすべて選ぶ。量が多くても、全部選ぶ(「全部やって」「3品全部」なら、料理をすべて)。
- 料理を頼む言い方は、必ず serve 系(最後まで作る)にする: 「作って」「作る」「最後まで」「出して」「提供して」「お願い」「やって」。
  cook 系(煮るところまで)は、「煮て」「鍋に入れて」「煮るところまで」のように、煮る工程で止めると明示されたときだけ。
  同じ料理について serve 系と cook 系の両方を選んではいけない(serve 系に煮る工程は含まれている)。
  「切って」「刻んで」= chop 系。
- 料理の指定は、言い回しに当てはまる料理をすべて選ぶ(曖昧ではない)。
  例: 「トマトに関するスープ」= トマトを使うスープ全部。「スープ」= スープ全部。「たまねぎのスープ」= たまねぎを使うスープ(1つしか無ければそれ)。
- 切る個数: 数が書かれていればその数(x1 / x2)。「両方」「全部」= x2。数が無く、その材料がこの回に1つしか要らなければ x1。
  数が無く、2つ要るなら ambiguous。
- 「私は〜をやる」のような自分の分担は無視し、AI への作業だけを選ぶ。「あとは任せる」「残りはお願い」は指定ではないので無視する。
'''


# ---- 候補 -> 一覧 -------------------------------------------------------------
def menu_of(candidates):
    """CSP の候補 [(display, payload), ...] を、LLM に見せる [(id, 説明)] にする。"""
    from agent.instruction_panel import card_label, card_action, card_steps
    out = []
    for display, p in candidates:
        if not isinstance(p, dict) or not p.get('verb'):
            continue
        verb, obj = p['verb'], p.get('obj', '')
        count, total = int(p.get('count') or 1), int(p.get('total') or 1)
        chain = (p.get('chains') or [p.get('chain') or []])[0]
        chained = len(chain) > 1
        label = card_label(verb, obj)
        if verb == 'chop':
            desc = f'{label}を{count}つ切る' + (f'(この回に要るのは{total}つ)' if total > 1 else '')
        elif verb in ('cook', 'mix'):
            desc = f'{label}を、材料を切って鍋に入れて煮るところまでやる'
        elif verb in ('serve', 'serve_salad', 'serve_juice', 'handover', 'serve_from_counter'):
            steps = card_steps(chain) if chained else ''
            desc = f'{label}を最後まで作る' + (f'({steps})' if steps else '')
        else:
            desc = f'{label}{card_action(verb, chained)}'
        out.append((str(display), desc))
    return out


def cap_check(task_ids, candidates=None):
    """上限を超えていないか。超えていれば理由を返す(無ければ None)。

    candidates が無ければ id の形(chop_<材料>_x<個数> / cook_... / serve_...)から数える。
    """
    by_id = {str(d): p for d, p in (candidates or [])}
    dishes = 0
    chops = 0
    for tid in task_ids:
        p = by_id.get(tid)
        if p is None:
            m = re.match(r'chop_.*_x(\d+)$', str(tid))
            if m:
                chops += int(m.group(1))
            elif str(tid).startswith('chop_'):
                chops += 1
            else:
                dishes += 1
            continue
        if p.get('verb') == 'chop':
            chops += int(p.get('count') or 1)
        elif p.get('verb'):
            dishes += 1
    if (MAX_DISHES is not None and dishes > MAX_DISHES) or \
            (MAX_CHOP_UNITS is not None and chops > MAX_CHOP_UNITS):
        return 'too_many'
    return None


def total_steps(candidates):
    """この回の工程の総数(候補の鎖に出てくる工程の集合の大きさ)。"""
    seen = set()
    for _d, p in (candidates or []):
        if not isinstance(p, dict):
            continue
        for g in (p.get('chains') or [p.get('chain') or []]):
            for c in g:
                seen.add(tuple(c))
    return len(seen)


def ai_steps(candidates, task_ids, exclude=()):
    """選んだ作業のうち、AI がやる工程の数(人の分と、除く工程は数えない)。"""
    by_id = {str(d): p for d, p in (candidates or [])}
    n = 0
    for tid in task_ids:
        p = by_id.get(tid)
        if not isinstance(p, dict):
            continue
        count = int(p.get('count') or 1)
        groups = list(p.get('chains') or [p.get('chain') or []])[:count]
        human = {tuple(c) for c in (p.get('human_ids') or [])}
        for g in groups:
            for c in g:
                if tuple(c) in human:
                    continue
                if len(c) > 2 and _step_kind(c[1]) in exclude:
                    continue
                n += 1
    return n


def size_check(task_ids, candidates=None, exclude=()):
    """AI の分担が 下限(MIN_AI_STEPS)〜上限(全体 x MAX_AI_SHARE)に収まるか。外れていれば理由。"""
    if not candidates:
        return None
    n = ai_steps(candidates, task_ids, exclude)
    total = total_steps(candidates)
    if n < MIN_AI_STEPS:
        return 'too_few'
    if total and n > total * MAX_AI_SHARE:
        return 'too_many'
    return None


# ---- 鍵と呼び出し -----------------------------------------------------------
def read_key(provider):
    env, fname = {'gemini': ('GOOGLE_API_KEY', 'google_api_key.txt'),
                  'openai': ('OPENAI_API_KEY', 'openai_api_key.txt'),
                  'anthropic': ('ANTHROPIC_API_KEY', 'anthropic_api_key.txt')}[provider]
    v = os.environ.get(env) or (os.environ.get('GEMINI_API_KEY') if provider == 'gemini' else None)
    if v:
        return v.strip()
    p = ROOT / fname
    if p.exists():
        return p.read_text(encoding='utf-8').strip() or None
    return None


def provider_of(model):
    if model.startswith('gemini'):
        return 'gemini'
    if model.startswith('claude'):
        return 'anthropic'
    return 'openai'


def _call_gemini(model, system, schema, text, timeout_s):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=read_key('gemini'),
                          http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    r = client.models.generate_content(
        model=model, contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            response_mime_type='application/json', response_schema=schema))
    return json.loads(r.text)


def _call_anthropic(model, system, schema, text, timeout_s):
    import anthropic
    client = anthropic.Anthropic(api_key=read_key('anthropic'), timeout=timeout_s)
    r = client.messages.create(
        model=model, max_tokens=400, temperature=0, system=system,
        messages=[{'role': 'user', 'content': text}],
        tools=[{'name': 'decide', 'description': '指示の解釈の結果', 'input_schema': schema}],
        tool_choice={'type': 'tool', 'name': 'decide'})
    block = next(b for b in r.content if getattr(b, 'type', '') == 'tool_use')
    return dict(block.input)


def _call_openai(model, system, schema, text, timeout_s):
    from openai import OpenAI
    client = OpenAI(api_key=read_key('openai'), timeout=timeout_s)
    kw = {}
    if not model.startswith(('o1', 'o3', 'o4', 'gpt-5', 'gpt-6')):
        kw['temperature'] = 0
    r = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
        response_format={'type': 'json_schema',
                         'json_schema': {'name': 'decide',
                                         'schema': dict(schema, additionalProperties=False),
                                         'strict': True}},
        **kw)
    return json.loads(r.choices[0].message.content)


CALLERS = {'gemini': _call_gemini, 'anthropic': _call_anthropic, 'openai': _call_openai}


def call_model(model, system, schema, text, timeout_s=25.0, retries=2):
    """1回呼ぶ。混雑(503)や割り当て超過(429)は短く待ってやり直す。"""
    caller = CALLERS[provider_of(model)]
    last = None
    for attempt in range(retries + 1):
        try:
            return caller(model, system, schema, text, timeout_s)
        except Exception as e:
            last = e
            msg = str(e)
            # 残高切れ(insufficient_quota)は待っても直らないので、やり直さず次のモデルへ
            retryable = ('429' in msg or '503' in msg or 'overloaded' in msg.lower())                 and 'insufficient_quota' not in msg and 'credit' not in msg.lower()
            if attempt < retries and retryable:
                m = re.search(r'retry in ([\d.]+)s', msg)
                time.sleep(min(float(m.group(1)) + 0.5 if m else 2.0 * (attempt + 1), 8.0))
                continue
            raise
    raise last


# ---- 解釈 ---------------------------------------------------------------------
def interpret(text, candidates, orders_ja, model=None, timeout_s=25.0):
    """文を解釈する。戻り値は辞書(必ず返す。失敗は decision='reject', reason='error')。

        decision      : 'accept' / 'reject'
        valid         : 有効な指示か(LLM の判定)
        ambiguous     : 曖昧か(LLM の判定)
        tasks         : 候補の id の一覧(accept のとき)
        reject_reason : invalid / ambiguous / too_many / error
        message       : 参加者に見せる短い文
        raw           : LLM の出力そのもの
        model, latency_s
    """
    models = [model] if model else [m for m in DEFAULT_MODELS if read_key(provider_of(m))]
    if not models:
        models = [DEFAULT_MODEL]
    menu = menu_of(candidates)
    ids = [i for i, _ in menu]
    system = system_prompt(menu, orders_ja)
    t0 = time.perf_counter()
    fail = {'decision': 'reject', 'valid': None, 'ambiguous': None, 'tasks': [], 'reject_reason': 'error',
            'message': REJECT_JA['error'],
            'raw': None, 'model': models[0]}
    errors = []
    raw = None
    for model in models:
        try:
            raw = call_model(model, system, schema_for(ids), str(text or '').strip(), timeout_s)
            break
        except Exception as e:
            errors.append('%s: %s: %s' % (model, type(e).__name__, str(e)[:160]))
    if raw is None:
        return dict(fail, error=' / '.join(errors), latency_s=round(time.perf_counter() - t0, 2))
    out = {'raw': raw, 'model': model, 'latency_s': round(time.perf_counter() - t0, 2), 'error': None}
    if not isinstance(raw, dict):
        out.update(fail)
        return out
    return dict(out, **normalize(raw, ids, candidates))


def normalize(raw, ids, candidates=None):
    """LLM の出力(valid / ambiguous / tasks / exclude_steps / message)を、受理・却下の形にする。"""
    tasks = [t for t in (raw.get('tasks') or []) if t in ids]
    tasks = list(dict.fromkeys(tasks))               # 重複は落とす(順は保つ)
    exclude = [e for e in (raw.get('exclude_steps') or []) if e in STEP_JA]
    valid = bool(raw.get('valid'))
    ambiguous = bool(raw.get('ambiguous'))
    message = str(raw.get('message') or '').strip()
    base = {'valid': valid, 'ambiguous': ambiguous, 'exclude': exclude}
    # 同じ作業の個数違い(x1 と x2)が両方選ばれていたら、個数が決まっていない
    # (実測: 「レタスサラダを作って」で、1つと2つの両方が選ばれた)
    stems = [re.sub(r'_x\d+$', '', t) for t in tasks]
    if len(set(stems)) < len(stems):
        ambiguous = True
        base['ambiguous'] = True
        message = '個数(1つ / 2つ)を入れて、書き直してください。'
    if not valid:
        return dict(base, decision='reject', tasks=[], reject_reason='invalid',
                    message=soft_message(message, 'invalid'))
    if ambiguous or not tasks:
        return dict(base, decision='reject', tasks=[], reject_reason='ambiguous',
                    message=soft_message(message, 'ambiguous'))
    cap = size_check(tasks, candidates, exclude) or cap_check(tasks, candidates)
    if cap:
        return dict(base, decision='reject', tasks=[], reject_reason=cap, message=REJECT_JA[cap])
    # 指示は「AI に今すぐやってほしい作業」なので、鎖の下の工程(切る・煮る)を人に
    # 残して上の工程だけ頼むことはできない(上を頼んだら下も AI がやる)。
    # 除けるのは一番上(提供)だけ(2026-10-06)
    if any(e != 'serve' for e in exclude):
        return dict(base, decision='reject', tasks=[], reject_reason='invalid',
                    message='料理を頼むときは、材料を切るところから任せる形で書き直してください'
                            '(煮るところまでで止めたいときは「煮て」と書けます)。')
    # 除く工程が、選んだ作業の全部(切るだけの指示で「切らないで」)なら、やることが無い。
    # 料理の指示は鎖(切る → 煮る → 提供)なので、提供を除いても切る・煮るが残る
    if exclude:
        by_id = {str(d): p for d, p in (candidates or [])}
        kinds = set()
        for t in tasks:
            p = by_id.get(t)
            chain = (p or {}).get('chain') or []
            if chain:
                kinds |= {_step_kind(c[1]) for c in chain if len(c) > 2}
            elif t.startswith('chop_'):
                kinds.add('chop')
            elif t.startswith('cook_'):
                kinds |= {'chop', 'cook'}
            else:
                kinds |= {'chop', 'cook', 'serve'}
        if all(k in exclude for k in kinds):
            return dict(base, decision='reject', tasks=[], reject_reason='ambiguous',
                        message='AI がやる作業が残りませんでした。AI にやってほしい作業を書き直してください。')
    return dict(base, decision='accept', tasks=tasks, reject_reason=None, message=message)


def expand_steps(candidates, task_ids, exclude=()):
    """選んだ候補を、AI がやる一連の作業(切る → 煮る → 提供)に広げて、日本語で返す。

    戻り値: [(候補の id, [工程の文, ...]), ...]。個数つきの候補は個数ぶん。
    exclude の工程(AI にやらせない)には「(あなた)」を添える。
    """
    from agent.instruction_panel import INGREDIENT_JP, card_label
    by_id = {str(d): p for d, p in candidates}
    out = []
    for tid in task_ids:
        p = by_id.get(tid)
        if not isinstance(p, dict):
            continue
        count = int(p.get('count') or 1)
        groups = list(p.get('chains') or [p.get('chain') or []])[:count]
        steps = []
        for g in groups:
            for fid in g:
                if len(fid) < 3:
                    continue
                verb, obj = str(fid[1]), str(fid[2])
                tag = '(あなた)' if _step_kind(verb) in exclude else ''
                if verb == 'chop':
                    steps.append(f'{INGREDIENT_JP.get(obj, obj)}を切る{tag}')
                elif verb in ('cook', 'mix'):
                    steps.append((f'{card_label(verb, obj)}を煮る' if verb == 'cook' else f'{card_label(verb, obj)}を混ぜる') + tag)
                elif verb == 'serve_salad':
                    steps.append(f'{card_label(verb, obj)}を盛って提供する{tag}')
                elif verb in ('serve', 'serve_juice', 'serve_from_counter'):
                    steps.append(f'{card_label(verb, obj)}を提供する{tag}')
                elif verb == 'handover':
                    steps.append(f'{card_label(verb, obj)}を渡す{tag}')
                else:
                    steps.append(f'{verb} {obj}')
        out.append((tid, steps))
    return out


def _step_kind(verb):
    """工程の種類(chop / cook / serve)。exclude_steps と突き合わせる。"""
    v = str(verb)
    if v == 'chop':
        return 'chop'
    if v in ('cook', 'mix'):
        return 'cook'
    return 'serve'


def expand_stages(candidates, task_ids, exclude=()):
    """選んだ候補を、画面の流れ図に出す段に分ける。

    戻り値: [(候補の id, [段, ...]), ...]。段 = 同時に進められる作業の一覧
    (切るは材料ごとに並列。計画の順序制約も、切る同士には順序を付けない)。
      切る: [レタスを切る, トマトを切る] -> 煮る: [X を煮る] -> 提供: [提供する]
    サラダは 盛り付ける -> 提供する の2段に分けて見せる。
    exclude の工程(人がやる)には「(あなた)」を添える。
    """
    from agent.instruction_panel import INGREDIENT_JP, card_label
    by_id = {str(d): p for d, p in candidates}
    out = []
    for tid in task_ids:
        p = by_id.get(tid)
        if not isinstance(p, dict):
            continue
        count = int(p.get('count') or 1)
        groups = list(p.get('chains') or [p.get('chain') or []])[:count]
        chop, cook, plate, serve = [], [], [], []
        for g in groups:
            for fid in g:
                if len(fid) < 3:
                    continue
                verb, obj = str(fid[1]), str(fid[2])
                tag = '(あなた)' if _step_kind(verb) in exclude else ''
                if verb == 'chop':
                    chop.append(f'{INGREDIENT_JP.get(obj, obj)}を切る{tag}')
                elif verb == 'cook':
                    cook.append(f'{card_label(verb, obj)}を煮る{tag}')
                elif verb == 'mix':
                    cook.append(f'{card_label(verb, obj)}を混ぜる{tag}')
                elif verb == 'serve_salad':
                    plate.append(f'{card_label(verb, obj)}を盛り付ける{tag}')
                    serve.append(f'提供する{tag}')
                elif verb in ('serve', 'serve_juice', 'serve_from_counter'):
                    serve.append(f'提供する{tag}')
                elif verb == 'handover':
                    serve.append(f'{card_label(verb, obj)}を渡す{tag}')
                else:
                    serve.append(f'{verb} {obj}')
        stages = [st for st in (chop, cook, plate, serve) if st]
        # 同じ段に同じ文が並ぶ(「提供する」が2つ)のは、1つにまとめる
        stages = [list(dict.fromkeys(st)) if st is not chop else st for st in stages]
        out.append((tid, stages))
    return out


def compose(candidates, task_ids, exclude=()):
    """複数の候補を、1つの指示(鎖の指示)にまとめる。1つだけで除く工程も無ければ、その候補のまま。

    個数つきの候補(たまねぎを1つ: かたまり2つのうち1つ)は、先頭から個数ぶんの
    かたまりに固定する。まとめた指示では「どのかたまりを AI がやるか」を
    ソルバーに選ばせない(選ばせると、別の作業のかたまりを選んでしまう)。
    exclude の工程(「切るのは私がやる」)は、鎖に残したまま人がやる工程にする
    (exclude_verbs。計画側はその工程を相手に割り当て、AI はそれを待つ)。
    """
    by_id = {str(d): (d, p) for d, p in candidates}
    picked = [by_id[t] for t in task_ids if t in by_id]
    exclude = [e for e in (exclude or []) if e in STEP_JA]
    if not picked:
        return None
    if len(picked) == 1 and not exclude:
        return picked[0]
    groups, fixed, humans, labels = [], [], [], []
    for display, p in picked:
        count = int(p.get('count') or 1)
        chains = list(p.get('chains') or [p.get('chain') or []])[:count]
        groups += chains
        fixed += list(p.get('fixed_task_ids') or [p.get('fixed_task_id')])[:count]
        humans += [c for c in (p.get('human_ids') or []) if any(tuple(c) == tuple(x) for ch in chains for x in ch)]
        labels.append(str(display))
    first = dict(picked[0][1])
    excl_verbs = sorted({str(c[1]) for g in groups for c in g if len(c) > 2 and _step_kind(c[1]) in exclude})
    if excl_verbs:
        humans = list(humans) + [c for g in groups for c in g if len(c) > 2 and str(c[1]) in excl_verbs]
    first.update({
        'chains': groups, 'count': len(groups), 'chain': [c for g in groups for c in g],
        'fixed_task_ids': [f for f in fixed if f], 'fixed_task_id': (fixed or [None])[0],
        'human_ids': humans, 'startable': all(p.get('startable', True) for _d, p in picked),
        'composite': [str(d) for d, _p in picked] if len(picked) > 1 else None,
        'exclude_verbs': excl_verbs,
    })
    label = '+'.join(labels) + ('(除く: ' + '/'.join(STEP_JA[e] for e in exclude) + ')' if exclude else '')
    return (label, first)
