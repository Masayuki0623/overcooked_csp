# -*- coding: utf-8 -*-
"""自然言語の指示を LLM に記号化させるとき、どのモデルが適切かを測る。

参加者の文(日本語)を、いまの指示の候補(動詞・対象・個数の id)の中から
選ばせる。候補に無いものや曖昧な文は却下(reject)させる。60 通りの文に
期待する答えを付けてあり、正答率と応答時間を出す。

    python tools/llm_instruction_bench.py                    # 鍵のある会社の既定モデル
    python tools/llm_instruction_bench.py --models gemini-3.8-flash,claude-haiku-4-5-20251001
    python tools/llm_instruction_bench.py --list             # 使えるモデルを列挙
    python tools/llm_instruction_bench.py --out results/llm_bench.json

鍵は環境変数(GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)か、
リポジトリ直下の google_api_key.txt / openai_api_key.txt / anthropic_api_key.txt
から読む(どれも git には入れない)。

候補の一覧は、パターン6の2回目の注文(たまねぎトマトのサラダ + たまねぎ
レタスのスープ + トマトレタスのスープ)を開始時点で CSP に出させたもの。
"""
import argparse
import json
import os
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- 候補(記号)。CSPAgent.get_instruction_candidates が出す id と同じ -------
MENU = [
    ('chop_lettuce_x1', 'レタスを1つ切る'),
    ('chop_lettuce_x2', 'レタスを2つ切る(この回に要るのは2つ)'),
    ('chop_onion_x1', 'たまねぎを1つ切る'),
    ('chop_onion_x2', 'たまねぎを2つ切る(この回に要るのは2つ)'),
    ('chop_tomato_x1', 'トマトを1つ切る'),
    ('chop_tomato_x2', 'トマトを2つ切る(この回に要るのは2つ)'),
    ('cook_lettuce_onionsoup', 'たまねぎレタスのスープを、材料を切って鍋に入れて煮るところまでやる'),
    ('cook_lettuce_tomatosoup', 'トマトレタスのスープを、材料を切って鍋に入れて煮るところまでやる'),
    ('serve_lettuce_onionsoup', 'たまねぎレタスのスープを最後まで作る(切る→煮る→皿に取って提供する)'),
    ('serve_lettuce_tomatosoup', 'トマトレタスのスープを最後まで作る(切る→煮る→皿に取って提供する)'),
    ('serve_salad_onion_tomatosalad', 'たまねぎトマトのサラダを最後まで作る(切る→皿に盛って提供する)'),
]
MENU_IDS = [m for m, _ in MENU]
ORDERS_JA = 'たまねぎトマトのサラダ、たまねぎレタスのスープ、トマトレタスのスープ(各1品)'
REJECT_REASONS = ['vague', 'ambiguous', 'nonexistent', 'too_many', 'not_for_ai']

# 説明文と出力の形は、本番と同じもの(agent/agent/instruction_nl.py)を使う。
sys.path.insert(0, str(ROOT / 'agent'))
from agent import instruction_nl as NL  # noqa: E402

SYSTEM = NL.system_prompt(MENU, ORDERS_JA)
SCHEMA = NL.schema_for(MENU_IDS)

# ---- 試験の文。expect は「受理ならば id の集合(別解があれば複数)」か、曖昧(AMB)・無効(INV) -
A = 'accept'
AMB = 'ambiguous'
INV = 'invalid'
SOUPS = ['serve_lettuce_onionsoup', 'serve_lettuce_tomatosoup']
ALL = SOUPS + ['serve_salad_onion_tomatosalad']
CASES = [
    # 単独・具体的
    ('たまねぎを1つ切って', A, [['chop_onion_x1']]),
    ('たまねぎを2つ切って', A, [['chop_onion_x2']]),
    ('オニオンを1個切って', A, [['chop_onion_x1']]),
    ('玉ねぎを1つ刻んで', A, [['chop_onion_x1']]),
    ('トマトを2個切ってください', A, [['chop_tomato_x2']]),
    ('レタスを1つ切る', A, [['chop_lettuce_x1']]),
    ('トマトを1つきって', A, [['chop_tomato_x1']]),
    ('レタスを全部切って', A, [['chop_lettuce_x2']]),
    ('トマトを両方切って', A, [['chop_tomato_x2']]),
    ('トマトレタススープを作って', A, [['serve_lettuce_tomatosoup']]),
    ('トマトとレタスのスープを最後まで作って', A, [['serve_lettuce_tomatosoup']]),
    ('レタス・トマトのスープを提供して', A, [['serve_lettuce_tomatosoup']]),
    ('たまねぎレタスのスープを出して', A, [['serve_lettuce_onionsoup']]),
    ('玉ねぎトマトサラダを作って', A, [['serve_salad_onion_tomatosalad']]),
    ('サラダを作って', A, [['serve_salad_onion_tomatosalad']]),
    ('サラダをお願い', A, [['serve_salad_onion_tomatosalad']]),
    ('トマトのサラダ', A, [['serve_salad_onion_tomatosalad']]),
    ('たまねぎのスープを作って', A, [['serve_lettuce_onionsoup']]),
    ('スープはたまねぎレタスのほうを作って', A, [['serve_lettuce_onionsoup']]),
    ('tomato lettuce soup を作って', A, [['serve_lettuce_tomatosoup']]),
    ('たまねぎレタスのスープを煮て', A, [['cook_lettuce_onionsoup']]),
    ('トマトレタススープを鍋に入れて', A, [['cook_lettuce_tomatosoup']]),
    ('レタストマトスープの調理をして', A, [['cook_lettuce_tomatosoup']]),
    ('スープの材料を煮るところまでやって。トマトレタスのやつ', A, [['cook_lettuce_tomatosoup']]),
    ('トマトを1つ切って、あとは任せる', A, [['chop_tomato_x1']]),
    # 言い回しで料理を絞る(当てはまる料理を全部)
    ('トマトに関するスープを作って', A, [['serve_lettuce_tomatosoup']]),
    ('トマトを使う料理を全部作って', A, [['serve_lettuce_tomatosoup', 'serve_salad_onion_tomatosalad']]),
    ('スープを作って', A, [SOUPS]),
    ('スープを煮て', A, [['cook_lettuce_onionsoup', 'cook_lettuce_tomatosoup']]),
    ('レタスのスープを作って', A, [SOUPS]),
    ('スープ2つとも作って', A, [SOUPS]),
    ('たまねぎレタスのスープとトマトレタススープを作って', A, [SOUPS]),
    ('サラダ以外を全部やって', A, [SOUPS]),
    ('全部あなたがやって', A, [ALL]),
    ('3品全部作って', A, [ALL]),
    ('たまねぎ、トマト、レタスを2つずつ切って', A, [['chop_onion_x2', 'chop_tomato_x2', 'chop_lettuce_x2']]),
    # 複合
    ('トマトレタススープを1つとたまねぎを1つ切って', A, [['serve_lettuce_tomatosoup', 'chop_onion_x1']]),
    ('トマトレタススープを作って、それからたまねぎを2つ切って', A, [['serve_lettuce_tomatosoup', 'chop_onion_x2']]),
    ('たまねぎ1つとトマト1つを切って', A, [['chop_onion_x1', 'chop_tomato_x1']]),
    ('たまねぎ2つとレタス1つを切って', A, [['chop_onion_x2', 'chop_lettuce_x1']]),
    ('サラダを作って、レタスも1つ切っておいて', A, [['serve_salad_onion_tomatosalad', 'chop_lettuce_x1']]),
    ('私はサラダをやるので、たまねぎレタスのスープをお願い', A, [['serve_lettuce_onionsoup']]),
    ('たまねぎを1つ切ったあと、トマトレタススープを煮て', A, [['chop_onion_x1', 'cook_lettuce_tomatosoup']]),
    # 工程の一部「だけ」= その段(前の工程を含む)
    ('スープの調理の部分だけをやって', A, [['cook_lettuce_onionsoup', 'cook_lettuce_tomatosoup']]),
    ('スープ料理を作るときの鍋に入れる作業だけをやって', A, [['cook_lettuce_onionsoup', 'cook_lettuce_tomatosoup']]),
    ('サラダの盛り付けだけやって', A, [['serve_salad_onion_tomatosalad']]),
    # 工程の一部を除く。除けるのは一番上(提供)だけ。下の工程(切る)を人に残す指示は無効
    ('トマトレタスのスープを作って。材料は切らないで', INV, None),
    ('たまねぎレタスのスープをお願い。切るのは私がやる', INV, None),
    ('サラダを作って。盛り付けと提供は私がやる', A, [['serve_salad_onion_tomatosalad']], ['serve']),
    # 曖昧: 作業が特定できない
    ('右で作業して', AMB, None),
    ('急いで', AMB, None),
    ('左側をお願い', AMB, None),
    ('手伝って', AMB, None),
    ('好きにやって', AMB, None),
    ('丁寧にやって', AMB, None),
    ('何か切って', AMB, None),
    ('鍋をよろしく', AMB, None),
    # 曖昧: 個数が決まらない(どの材料も2つ要る)
    ('たまねぎを切って', AMB, None),
    ('トマトをきって', AMB, None),
    ('野菜を切って', AMB, None),
    ('たまねぎとトマトを切って', AMB, None),
    # 無効: 注文に無い / AI への作業が無い
    ('ジュースを作って', INV, None),
    ('にんじんを切って', INV, None),
    ('たまねぎを3つ切って', INV, None),
    ('フルサラダを作って', INV, None),
    ('トマトレタスのサラダを作って', INV, None),
    ('こんにちは', INV, None),
    ('私がスープをやります', INV, None),
    ('どっちを先にやればいい?', INV, None),
]


# ---- 鍵 -------------------------------------------------------------------
def read_key(env_name, file_name):
    v = os.environ.get(env_name)
    if v:
        return v.strip()
    p = ROOT / file_name
    if p.exists():
        return p.read_text(encoding='utf-8').strip() or None
    return None


KEYS = {
    'gemini': read_key('GOOGLE_API_KEY', 'google_api_key.txt') or read_key('GEMINI_API_KEY', 'google_api_key.txt'),
    'openai': read_key('OPENAI_API_KEY', 'openai_api_key.txt'),
    'anthropic': read_key('ANTHROPIC_API_KEY', 'anthropic_api_key.txt'),
}


def provider_of(model):
    if model.startswith('gemini'):
        return 'gemini'
    if model.startswith('claude'):
        return 'anthropic'
    return 'openai'


# ---- 各社の呼び出し。戻り値は (dict, 使ったトークンのメモ) ----------------------
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
    return json.loads(r.text), {
        'in': getattr(usage, 'prompt_token_count', None),
        'out': getattr(usage, 'candidates_token_count', None),
        'thinking': getattr(usage, 'thoughts_token_count', None)}


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


CALLERS = {'gemini': call_gemini, 'anthropic': call_anthropic, 'openai': call_openai}


def list_models():
    out = {}
    if KEYS['gemini']:
        from google import genai
        c = genai.Client(api_key=KEYS['gemini'])
        out['gemini'] = sorted(m.name.replace('models/', '') for m in c.models.list()
                               if 'generateContent' in (getattr(m, 'supported_actions', None) or [])
                               and m.name.replace('models/', '').startswith('gemini'))
    if KEYS['anthropic']:
        import anthropic
        c = anthropic.Anthropic(api_key=KEYS['anthropic'])
        out['anthropic'] = sorted(m.id for m in c.models.list(limit=100))
    if KEYS['openai']:
        from openai import OpenAI
        c = OpenAI(api_key=KEYS['openai'])
        out['openai'] = sorted(m.id for m in c.models.list() if m.id.startswith(('gpt', 'o')))
    return out


# ---- 採点 --------------------------------------------------------------------
def judge(expect_kind, expect, got, expect_exclude=None):
    """(受理/曖昧/無効の区分が合っているか, 中身まで合っているか, 短い説明)"""
    if not isinstance(got, dict):
        return False, False, '形式が違う'
    # 本番と同じ後処理(id の検査・重複落とし・上限)を通す
    n = NL.normalize(got, MENU_IDS)
    kind = ('accept' if n['decision'] == 'accept'
            else 'invalid' if n['reject_reason'] == 'invalid' else 'ambiguous')
    if expect_kind == A:
        if kind != 'accept':
            return False, False, '%sと判定(valid=%s ambiguous=%s tasks=%s)' % (
                kind, got.get('valid'), got.get('ambiguous'), got.get('tasks'))
        tasks = set(n['tasks'])
        ok = any(tasks == set(alt) for alt in expect)
        if ok and set(n.get('exclude') or []) != set(expect_exclude or []):
            return True, False, '除く工程が違う: %s(期待 %s)' % (n.get('exclude'), expect_exclude)
        return True, ok, '' if ok else '選んだ: %s' % sorted(tasks)
    if kind == 'accept':
        return False, False, '受理された: %s' % sorted(n['tasks'])
    ok = kind == expect_kind
    return ok, ok, '' if ok else '%sと判定' % kind


def run_model(model, workers=4, repeat=1, only=''):
    caller = CALLERS[provider_of(model)]
    rows = []

    def one(idx_case):
        idx, case = idx_case
        text, kind, expect = case[0], case[1], case[2]
        expect_exclude = case[3] if len(case) > 3 else None
        # 混雑(503)や割り当て超過(429。無料枠は1分あたりの回数が小さい)は、
        # 言われた秒数だけ待ってやり直す。応答時間には待ちを含めない。
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
        d_ok, full_ok, why = judge(kind, expect, got, expect_exclude) if got is not None else (False, False, err)
        return {'i': idx, 'text': text, 'kind': kind, 'expect': expect, 'got': got,
                'decision_ok': d_ok, 'ok': full_ok, 'why': why, 'latency_s': round(dt, 2), 'usage': usage,
                'error': err}

    jobs = [(i, c) for _ in range(repeat) for i, c in enumerate(CASES) if not only or only in c[0]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
    rows.sort(key=lambda r: r['i'])
    return rows


def summarize(model, rows):
    n = len(rows)
    acc_full = sum(r['ok'] for r in rows)
    acc_dec = sum(r['decision_ok'] for r in rows)
    errs = sum(1 for r in rows if r['error'])
    lat = [r['latency_s'] for r in rows if not r['error']]
    acc_rows = [r for r in rows if r['kind'] == A]
    rej_rows = [r for r in rows if r['kind'] != A]
    return {
        'model': model, 'n': n, 'errors': errs,
        'ok': acc_full, 'decision_ok': acc_dec,
        'accept_ok': sum(r['ok'] for r in acc_rows), 'accept_n': len(acc_rows),
        'reject_ok': sum(r['decision_ok'] for r in rej_rows), 'reject_n': len(rej_rows),
        'reject_reason_ok': sum(r['ok'] for r in rej_rows),
        'latency_median': round(st.median(lat), 2) if lat else None,
        'latency_p90': round(sorted(lat)[int(len(lat) * 0.9) - 1], 2) if len(lat) >= 10 else None,
        'latency_max': round(max(lat), 2) if lat else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--only', default='', help='文にこの語を含む件だけ測る')
    args = ap.parse_args()
    if args.list:
        for prov, ms in list_models().items():
            print('==', prov)
            for m in ms:
                print('  ', m)
        return
    models = [m for m in args.models.split(',') if m]
    if not models:
        if KEYS['gemini']:
            models += ['gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-3.1-pro-preview', 'gemini-2.5-flash']
        if KEYS['anthropic']:
            models += ['claude-haiku-4-5-20251001', 'claude-sonnet-5-5']
        if KEYS['openai']:
            models += ['gpt-5-mini', 'gpt-5']
    results = {}
    for m in models:
        if not KEYS.get(provider_of(m)):
            print(f'[{m}] 鍵が無いので飛ばします')
            continue
        print(f'[{m}] {len(CASES)} 件 x {args.repeat} を測ります...', flush=True)
        rows = run_model(m, args.workers, args.repeat, args.only)
        s = summarize(m, rows)
        results[m] = {'summary': s, 'rows': rows}
        print('  正答 %d/%d (受理/却下の判断 %d, 受理の中身 %d/%d, 却下 %d/%d うち理由も一致 %d) '
              '応答 中央値 %.2fs 90%% %.2fs 最大 %.2fs 失敗 %d' % (
                  s['ok'], s['n'], s['decision_ok'], s['accept_ok'], s['accept_n'],
                  s['reject_ok'], s['reject_n'], s['reject_reason_ok'],
                  s['latency_median'] or 0, s['latency_p90'] or 0, s['latency_max'] or 0, s['errors']), flush=True)
        for r in rows:
            if not r['ok']:
                print('    x %-40s 期待=%s | %s' % (r['text'], r['kind'] if r['kind'] != A else r['expect'][0], r['why']))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
        print('書き出し:', args.out)


if __name__ == '__main__':
    main()
