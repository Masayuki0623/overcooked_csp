"""文章の指示の解釈を、いまの規則(2026-10-06)で測る(v2)。

候補(記号)は本番と同じく CSPAgent.get_instruction_candidates から作り、後処理も
本番の instruction_nl.normalize(候補つき: 量の上限・下限、同名の個数、除く工程)を
通す。判定の種類(受理 / 却下の種類)まで採点する。

規則(v1 からの変更):
  - 量: エージェントの工程が 2 つ未満は too_few、全体の半分(この注文では 5)を超えると too_many
  - 「〜だけ」はその段(前の工程を含む)。除けるのは提供だけ(切るを人に残すのは exclude_lower)
  - 材料の言い換え(玉ねぎ / オニオン / onion)、料理名の材料の順番は問わない
  - 却下は種類(pattern)つき

    python tools/llm_instruction_bench2.py --models gpt-5.4-mini,gemini-3.5-flash-lite
    python tools/llm_instruction_bench2.py --list        # 文の一覧
"""
import argparse
import json
import os
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from agent import instruction_nl as NL  # noqa: E402
from llm_instruction_bench import KEYS, provider_of  # noqa: E402

RECIPES = ('OnionTomatoSalad', 'OnionLettuceSoup', 'TomatoLettuceSoup')
ORDERS_JA = 'たまねぎトマトのサラダ、たまねぎレタスのスープ、トマトレタスのスープ(各1品)'


def real_candidates():
    """本番と同じ候補(鎖つき)。地図はリング(鍋2つ、野菜のみ)。"""
    from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
    from gym_cooking.utils.replay import Replay
    from agent.executor.low import EnvState
    from agent.myagent.CSPAgent import CSPAgent
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg', order_recipes=RECIPES))
    env.reset()
    i = env.get_ai_info()
    st_ = EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0, order=i['order_scheduler'],
                   event_history=i['event_history'], time=i['current_time'], chg_grid=i['chg_grid'])
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    return ai.get_instruction_candidates(deepcopy(st_))


CANDS = real_candidates()
MENU = NL.menu_of(CANDS)
MENU_IDS = [m for m, _ in MENU]
SYSTEM = NL.system_prompt(MENU, ORDERS_JA)
SCHEMA = NL.schema_for(MENU_IDS)

# ---- 試験の文 ------------------------------------------------------------------
# (文, 'A', [受理される id の集合(別解があれば複数)], 除く工程)  受理
# (文, 'R', {却下の種類, ...})                                   却下(種類のどれかなら正解)
A, R = 'accept', 'reject'
SAL, OL, LT = 'serve_salad_onion_tomatosalad', 'serve_lettuce_onionsoup', 'serve_lettuce_tomatosoup'
COL, CLT = 'cook_lettuce_onionsoup', 'cook_lettuce_tomatosoup'
CASES = [
    # --- 受理: 料理を最後まで(serve 系) ---
    ('トマトレタススープを作って', A, [[LT]]),
    ('たまねぎレタススープを最後まで作って', A, [[OL]]),
    ('レタス玉ねぎスープをお願い', A, [[OL]]),
    ('オニオンとトマトのサラダを作って', A, [[SAL]]),
    ('サラダを提供して', A, [[SAL]]),
    ('サラダをやって', A, [[SAL]]),
    ('サラダを作ってください', A, [[SAL]]),
    ('トマトとレタスのスープを出して', A, [[LT]]),
    ('lettuce tomato soup を作って', A, [[LT]]),
    ('トマトを使うスープを作って', A, [[LT]]),
    ('トマト入りのスープを最後まで', A, [[LT]]),
    ('サラダの盛り付けだけやって', A, [[SAL]]),
    ('たまねぎレタススープ、提供まで', A, [[OL]]),
    ('スープを1品、トマトが入ってるほうを作って', A, [[LT]]),
    # --- 受理: 煮るところまで(cook 系) ---
    ('トマトレタススープを煮て', A, [[CLT]]),
    ('たまねぎレタススープを鍋に入れて', A, [[COL]]),
    ('レタス玉ねぎスープの調理だけして', A, [[COL]]),
    ('トマトレタススープの鍋に入れる作業だけやって', A, [[CLT]]),
    ('たまねぎレタススープを煮るところまで', A, [[COL]]),
    ('たまねぎが入っているスープを煮て', A, [[COL]]),
    # --- 受理: 切る(2 工程以上) ---
    ('たまねぎを2つ切って', A, [['chop_onion_x2']]),
    ('レタスを両方切って', A, [['chop_lettuce_x2']]),
    ('トマトを全部切って', A, [['chop_tomato_x2']]),
    ('たまねぎ1つとトマト1つを切って', A, [['chop_onion_x1', 'chop_tomato_x1']]),
    ('オニオンを2個刻んで', A, [['chop_onion_x2']]),
    ('レタス1つとたまねぎ1つ', A, [['chop_lettuce_x1', 'chop_onion_x1']]),
    ('玉ねぎを2つ切っておいて', A, [['chop_onion_x2']]),
    ('トマトを1つとレタスを1つ切って', A, [['chop_tomato_x1', 'chop_lettuce_x1']]),
    ('レタスを2つ刻んで', A, [['chop_lettuce_x2']]),
    ('トマト2個切って', A, [['chop_tomato_x2']]),
    # --- 受理: 組み合わせ(合計 5 工程まで) ---
    ('トマトレタススープを作って、たまねぎも1つ切って', A, [[LT, 'chop_onion_x1']]),
    ('サラダを作って、レタスを2つ切って', A, [[SAL, 'chop_lettuce_x2']]),
    ('たまねぎレタススープを煮て、トマトを2つ切って', A, [[COL, 'chop_tomato_x2']]),
    ('私はサラダをやるので、トマトレタススープをお願い', A, [[LT]]),
    ('トマトレタススープを作って。あとは任せる', A, [[LT]]),
    # --- 受理: 提供だけ人がやる ---
    ('サラダを作って。盛り付けと提供は私がやる', A, [[SAL]], ['serve']),
    ('トマトレタススープを作って、提供は私がやる', A, [[LT]], ['serve']),
    # --- 却下: 少なすぎ(1 工程) ---
    ('たまねぎを1つ切って', R, {'too_few'}),
    ('トマトを1個だけ切って', R, {'too_few'}),
    ('レタスを1つ切って', R, {'too_few'}),
    # --- 却下: 多すぎ(全体の半分 = 5 工程を超える) ---
    ('スープを両方作って', R, {'too_many'}),
    ('全部やって', R, {'too_many'}),
    ('全部あなたがやって', R, {'too_many'}),
    ('サラダとトマトレタススープを作って', R, {'too_many'}),
    ('スープを煮て', R, {'too_many'}),
    ('トマトレタススープを作って、たまねぎを2つ切って', R, {'too_many'}),
    ('スープを作って', R, {'too_many'}),
    # --- 却下: 個数が決まらない(その材料は 2 つ要る) ---
    ('たまねぎを切って', R, {'count_missing'}),
    ('トマトを切って', R, {'count_missing'}),
    ('野菜を切って', R, {'count_missing', 'no_task'}),
    ('レタスとトマトを切って', R, {'count_missing'}),
    ('何か切って', R, {'count_missing', 'no_task'}),
    # --- 却下: 作業が特定できない ---
    ('右で作業して', R, {'no_task'}),
    ('急いで', R, {'no_task'}),
    ('手伝って', R, {'no_task'}),
    ('好きにやって', R, {'no_task'}),
    ('鍋をよろしく', R, {'no_task'}),
    ('左側をお願い', R, {'no_task'}),
    # --- 却下: この回の注文に無い ---
    ('ジュースを作って', R, {'not_in_orders'}),
    ('にんじんを切って', R, {'not_in_orders'}),
    ('全部のせサラダを作って', R, {'not_in_orders'}),
    ('たまねぎトマトスープを作って', R, {'not_in_orders'}),
    ('レタスのサラダを作って', R, {'not_in_orders'}),
    ('りんごを切って', R, {'not_in_orders'}),
    # --- 却下: 要る数より多い ---
    ('たまねぎを3つ切って', R, {'count_over'}),
    ('レタスを3つ切って', R, {'count_over'}),
    # --- 却下: 指示ではない ---
    ('こんにちは', R, {'not_instruction'}),
    ('私がスープをやります', R, {'not_instruction'}),
    ('どっちを先にやればいい?', R, {'not_instruction'}),
    ('ありがとう', R, {'not_instruction'}),
    ('いい感じだね', R, {'not_instruction'}),
    # --- 却下: 下の工程(切る)を人に残す ---
    ('トマトレタススープを作って。材料は切らないで', R, {'exclude_lower'}),
    ('たまねぎレタススープをお願い。切るのは私がやる', R, {'exclude_lower'}),
    ('サラダを作って。切るのはこっちでやるから盛り付けから', R, {'exclude_lower'}),
]


# ---- 各社の呼び出し ------------------------------------------------------------
def call_gemini(model, text):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=KEYS['gemini'])
    r = client.models.generate_content(
        model=model, contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0,
            response_mime_type='application/json', response_schema=SCHEMA))
    usage = getattr(r, 'usage_metadata', None)
    return json.loads(r.text), {'in': getattr(usage, 'prompt_token_count', None),
                                'out': getattr(usage, 'candidates_token_count', None)}


def call_openai(model, text):
    from openai import OpenAI
    client = OpenAI(api_key=KEYS['openai'])
    kw = {}
    if not model.startswith(('o1', 'o3', 'o4', 'gpt-5')):
        kw['temperature'] = 0
    r = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
        response_format={'type': 'json_schema',
                         'json_schema': {'name': 'decide', 'schema': dict(SCHEMA, additionalProperties=False),
                                         'strict': True}},
        **kw)
    u = r.usage
    return json.loads(r.choices[0].message.content), {'in': u.prompt_tokens, 'out': u.completion_tokens}


def call_anthropic(model, text):
    import anthropic
    client = anthropic.Anthropic(api_key=KEYS['anthropic'])
    r = client.messages.create(
        model=model, max_tokens=400, temperature=0, system=SYSTEM,
        messages=[{'role': 'user', 'content': text}],
        tools=[{'name': 'decide', 'description': '指示の解釈の結果', 'input_schema': SCHEMA}],
        tool_choice={'type': 'tool', 'name': 'decide'})
    block = next(b for b in r.content if getattr(b, 'type', '') == 'tool_use')
    return dict(block.input), {'in': r.usage.input_tokens, 'out': r.usage.output_tokens}


CALLERS = {'gemini': call_gemini, 'anthropic': call_anthropic, 'openai': call_openai}


# ---- 採点 ----------------------------------------------------------------------
def judge(case, got):
    """(受理/却下の判定が合っているか, 中身(id と除く工程 / 却下の種類)まで合っているか, 説明)"""
    if not isinstance(got, dict):
        return False, False, '形式が違う'
    n = NL.normalize(got, MENU_IDS, CANDS)      # 本番と同じ後処理(量の上限・下限も)
    kind = A if n['decision'] == 'accept' else R
    if case[1] == A:
        if kind != A:
            return False, False, '却下(%s): %s' % (n.get('pattern'), (n.get('message') or '')[:40])
        tasks = set(n['tasks'])
        ok = any(tasks == set(alt) for alt in case[2])
        if ok and set(n.get('exclude') or []) != set(case[3] if len(case) > 3 else []):
            return True, False, '除く工程が違う: %s' % n.get('exclude')
        return True, ok, '' if ok else '選んだ: %s' % sorted(tasks)
    if kind == A:
        return False, False, '受理された: %s' % sorted(n['tasks'])
    ok = n.get('pattern') in case[2]
    return True, ok, '' if ok else '種類が違う: %s(期待 %s)' % (n.get('pattern'), '/'.join(sorted(case[2])))


def run_model(model, workers=4, repeat=1, only=''):
    caller = CALLERS[provider_of(model)]
    rows = []

    def one(idx_case):
        idx, case = idx_case
        text = case[0]
        got, usage, err, dt = None, {}, None, 0.0
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                got, usage = caller(model, text)
                err = None
                dt = time.perf_counter() - t0
                break
            except Exception as e:
                msg = str(e)
                err = '%s: %s' % (type(e).__name__, msg[:200])
                if '429' in msg or '503' in msg or 'RESOURCE_EXHAUSTED' in msg or 'overloaded' in msg.lower():
                    m = re.search(r'retry in ([\d.]+)s', msg)
                    wait = float(m.group(1)) + 1.0 if m else 5.0 * (attempt + 1)
                    time.sleep(min(wait, 70.0))
                    continue
                break
        d_ok, full_ok, why = judge(case, got) if got is not None else (False, False, err)
        return {'i': idx, 'text': text, 'kind': case[1], 'expect': list(case[2]) if case[1] == R else case[2],
                'got': got, 'decision_ok': d_ok, 'ok': full_ok, 'why': why, 'latency_s': round(dt, 2),
                'usage': usage, 'error': err}

    jobs = [(i, c) for _ in range(repeat) for i, c in enumerate(CASES) if not only or only in c[0]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
    rows.sort(key=lambda r: r['i'])
    return rows


def summarize(model, rows):
    lat = [r['latency_s'] for r in rows if not r['error']]
    acc = [r for r in rows if r['kind'] == A]
    rej = [r for r in rows if r['kind'] == R]
    return {
        'model': model, 'n': len(rows), 'errors': sum(1 for r in rows if r['error']),
        'ok': sum(r['ok'] for r in rows), 'decision_ok': sum(r['decision_ok'] for r in rows),
        'accept_ok': sum(r['ok'] for r in acc), 'accept_n': len(acc),
        'reject_decision_ok': sum(r['decision_ok'] for r in rej), 'reject_ok': sum(r['ok'] for r in rej),
        'reject_n': len(rej),
        'latency_median': round(st.median(lat), 2) if lat else None,
        'latency_p90': round(sorted(lat)[int(len(lat) * 0.9) - 1], 2) if len(lat) >= 10 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='gpt-5.4-mini')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    if args.list:
        for i, c in enumerate(CASES):
            print('%2d %-6s %-34s %s' % (i + 1, c[1], c[0], c[2] if c[1] == A else '/'.join(sorted(c[2]))))
        print('候補:', MENU_IDS)
        return 0
    results = {}
    for m in [x for x in args.models.split(',') if x]:
        if not KEYS.get(provider_of(m)):
            print(f'[{m}] 鍵が無いので飛ばします')
            continue
        workers = 1 if provider_of(m) == 'gemini' else args.workers     # 無料枠は 1 分あたりの回数が小さい
        print(f'[{m}] {len(CASES)} 件 x {args.repeat} を測ります...', flush=True)
        rows = run_model(m, workers, args.repeat, args.only)
        s = summarize(m, rows)
        results[m] = {'summary': s, 'rows': rows}
        print('  正答 %d/%d (受理/却下の判定 %d/%d | 受理の中身 %d/%d | 却下: 判定 %d/%d, 種類まで %d/%d) '
              '応答 中央値 %ss 90%% %ss 失敗 %d' % (
                  s['ok'], s['n'], s['decision_ok'], s['n'], s['accept_ok'], s['accept_n'],
                  s['reject_decision_ok'], s['reject_n'], s['reject_ok'], s['reject_n'],
                  s['latency_median'], s['latency_p90'], s['errors']))
        for r in rows:
            if not r['ok']:
                print('    x %-34s 期待=%s | %s' % (r['text'], r['expect'], r['why']))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
        print('書き出し:', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
