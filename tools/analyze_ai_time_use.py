"""リプレイから、AI が「指示された作業」と「それ以外の作業」に使った時間を測る。

    python tools/analyze_ai_time_use.py                  # 有効な参加者(analyze_pattern7.VALID)
    python tools/analyze_ai_time_use.py --show p32:2     # 1 回分を時刻つきで並べる

数え方(0.2 秒ごとの AI の 1 手を、どれか 1 つに振り分ける):
  止まっている : その手が「何もしない」
  指示の作業   : 指示に入っている材料・料理を持っている、またはそれを取りに行く途中
  ほかの作業   : それ以外の物を持っている、またはそれを取りに行く途中
手ぶらで動いている手は、次に手に取る物(またはまな板で切る物)の側に入れる。

「指示の作業」の中身:
  - 切る指示(chop_X_xN): 材料 X を N 個ぶん。AI が自分で切って取り上げた X が N 個に
    なるまでに手を付けた X は、全部指示の作業(人に取られて切り直した分も含む)
  - 料理の指示(serve_/cook_/serve_salad_): その料理の材料を 1 つずつ切る・合わせる・
    煮る・皿にのせる・出す
  - 人が切った物を AI がまな板からどかしただけ、などはほかの作業
見る範囲は、指示を受けてから(0 秒)指示が終わるまで:
  切る指示 = AI が必要な数を切り終えた時刻、料理の指示 = その料理が出た時刻(誰が出しても)、
  煮る指示 = 鍋に入った時刻。終わらなかった回はゲームの終わりまで。
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
os.chdir(ROOT)

import analyze_pattern7 as A  # noqa: E402
import replay_trace as R  # noqa: E402

INGS = ('lettuce', 'tomato', 'onion')


def parse_instruction(task):
    """'chop_lettuce_x2+chop_tomato_x2' -> (切る数, 料理の材料, 料理の種類, 終わり方)"""
    need = Counter()
    explicit = Counter()       # 「切って」で数を言われた分
    dish, kind, end = None, None, 'chop'
    for tok in str(task or '').split('+'):
        m = re.match(r'chop_([a-z]+)(?:_x(\d+))?$', tok)
        if m:
            need[m.group(1)] += int(m.group(2) or 1)
            explicit[m.group(1)] += int(m.group(2) or 1)
            continue
        m = re.match(r'(serve_salad|serve|cook)_([a-z_]+?)(soup|salad)$', tok)
        if m:
            dish = tuple(sorted(m.group(2).strip('_').split('_')))
            kind = m.group(3)
            end = 'cook' if m.group(1) == 'cook' else 'deliver'
            for ing in dish:
                need[ing] = max(need[ing], 1)
    return need, explicit, dish, kind, end


def parts_of(name):
    """'ChoppedLettuce-ChoppedOnion-Plate' -> [('Chopped','lettuce'), ...], 皿つきか"""
    if not name:
        return [], False
    ps = str(name).split('-')
    plate = 'Plate' in ps
    out = []
    for p in ps:
        m = re.match(r'(Fresh|Chopped|Cooked)([A-Z][a-z]+)$', p)
        if m:
            out.append((m.group(1), m.group(2).lower()))
    return out, plate


def analyze(path):
    info = R.load(path)
    sel = info.get('web_selection') or {}
    acc = info.get('accepted') or {}
    need, explicit, dish, kind, end_kind = parse_instruction(acc.get('task'))
    env = R.rebuild(sel)
    ai = env.sim_agents[0]
    steps = [h for h in info['his'] if h['name'] == 'env.step']

    # まず全部流して、AI の手ごとの持ち物と出来事を集める
    frames = []           # (時刻の始まり, 長さ, 何もしないか, 持ち物の名前, その手の出来事[])
    seen = set()
    events = []
    t = 0.0
    for h in steps:
        dt = h['args'].get('passed_time', 0.2)
        act = tuple(h['args']['action_dict'].get(ai.name) or (0, 0))
        # この手のあいだ持っていた物(置く・出す手は、置く前の物の作業に入れる)
        hold_before = getattr(ai.holding, 'full_name', None)
        acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0)) for a in env.sim_agents}
        env.step(acts, passed_time=dt)
        evs = []
        allev = []
        for ev in list(env._event_history):
            if id(ev) in seen:
                continue
            seen.add(id(ev))
            name = str(getattr(ev, 'event', ''))
            if name == 'Move':
                continue
            by_ai = getattr(ev, 'playerA', None) == ai.name
            events.append((float(ev.time), by_ai, name))
            allev.append(name)
            if by_ai:
                evs.append(name)
        hold_after = getattr(ai.holding, 'full_name', None)
        frames.append((t, dt, act == (0, 0), hold_before or hold_after, evs, allev))
        t = env.current_time
    game_end = t

    # 指示の終わり
    end_t = None
    dish_set = set(dish or ())
    # 指示のスープが煮えている間(この間のほかの作業は、どの d でも許されている)
    cook_t = None
    if kind == 'soup':
        for et, by_ai, name in events:
            if name.startswith('Cook_'):
                ps, _ = parts_of(name[len('Cook_'):])
                if {i for _, i in ps} == dish_set:
                    cook_t = et
                    break
    cook_s = float(getattr(sys.modules.get('gym_cooking.utils.config'), 'COOKING_TIME_SECONDS', 15))
    for et, by_ai, name in events:
        if end_kind == 'deliver' and name.startswith('Deliver_'):
            ps, _ = parts_of(name[len('Deliver_'):])
            if {i for _, i in ps} == dish_set:
                end_t = et
                break
        if end_kind == 'cook' and name.startswith('Cook_'):
            ps, _ = parts_of(name[len('Cook_'):])
            if {i for _, i in ps} == dish_set:
                end_t = et
                break

    # 材料ごとの「ひとまとまり」(AI が生の材料を取ってから、切った物を置くまで)
    retrieved = Counter()      # AI が自分で切って取り上げた数
    chopped_by_ai = Counter()  # いま手がけている材料を AI が切ったか
    unit_cls = {}              # 材料 -> いまの単位の分類
    labels = []                # 手ごとの分類(手ぶらは後で埋める)
    chop_done_t = None
    supplied = set()           # 指示の料理に、もう誰かが入れた材料
    for (t0, dt, idle, hold, evs, allev) in frames:
        cls = None
        for e in allev:
            m = re.match(r'(Assemble|Cook)_(.+)$', e)
            if m and dish_set:
                ps, _ = parts_of(m.group(2))
                got = {i for _, i in ps}
                if len(got) >= 2 and got <= dish_set:
                    supplied |= got
        for e in evs:
            m = re.match(r'Pickup_Fresh([A-Z][a-z]+)_from_', e)
            if m:
                ing = m.group(1).lower()
                # 数を言われた「切る」はその数まで。料理の材料は、まだ料理に入って
                # いなければ 1 つ(人が先に入れたら、その材料はもう指示の分ではない)
                if retrieved[ing] < explicit.get(ing, 0):
                    unit_cls[ing] = 'instr'
                elif ing in dish_set and ing not in supplied and retrieved[ing] < 1:
                    unit_cls[ing] = 'instr'
                else:
                    unit_cls[ing] = 'other'
                chopped_by_ai[ing] = False
            m = re.match(r'Chop_Fresh([A-Z][a-z]+)$', e)
            if m:
                chopped_by_ai[m.group(1).lower()] = True
            m = re.match(r'Pickup_Chopped([A-Z][a-z]+)_from_Cutboard$', e)
            if m:
                ing = m.group(1).lower()
                if chopped_by_ai.get(ing) and unit_cls.get(ing) == 'instr':
                    retrieved[ing] += 1
                elif not chopped_by_ai.get(ing):
                    unit_cls[ing] = 'other'       # 人が切った物を動かしただけ
                chopped_by_ai[ing] = False
        if end_kind == 'chop' and chop_done_t is None and need and all(
                retrieved[i] >= n for i, n in need.items()):
            chop_done_t = t0 + dt
        # 持ち物で分類
        ps, plate = parts_of(hold)
        if ps:
            ings = {i for _, i in ps}
            if len(ps) == 1 and ps[0][0] in ('Fresh', 'Chopped') and not plate:
                cls = unit_cls.get(ps[0][1], 'other')
            elif dish_set and ings <= dish_set and (
                    kind == 'soup' or all(s == 'Chopped' for s, _ in ps)):
                cls = 'instr'
            else:
                cls = 'other'
        elif hold == 'Plate':
            cls = 'plate'          # 次に何をのせるかで決める
        # まな板で切っている手(手ぶら)
        for e in evs:
            m = re.match(r'Chop_Fresh([A-Z][a-z]+)$', e)
            if m:
                cls = unit_cls.get(m.group(1).lower(), 'other')
        labels.append('idle' if idle else cls)
    if end_kind == 'chop':
        end_t = chop_done_t

    # 手ぶら(None)と皿だけ('plate')は、次に分類が付く手に合わせる
    nxt = 'other'
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] in ('instr', 'other'):
            nxt = labels[i]
        elif labels[i] in (None, 'plate'):
            labels[i] = nxt
    window = end_t if end_t is not None else game_end
    tot = Counter()
    tot_all = Counter()
    episodes = []          # ほかの作業のまとまり (始まり, 終わり, 持ち物)
    cur = None
    for (t0, dt, idle, hold, evs, _a), lab in zip(frames, labels):
        tot_all[lab] += dt
        if t0 < window:
            tot[lab] += dt
            if lab == 'other' and cook_t is not None and cook_t <= t0 < cook_t + cook_s:
                tot['other_wait'] += dt
            if lab == 'other':
                if cur and abs(cur[1] - t0) < 1e-6:
                    cur[1] = t0 + dt
                    if hold:
                        cur[2].add(hold)
                else:
                    cur = [t0, t0 + dt, {hold} if hold else set()]
                    episodes.append(cur)
            elif lab != 'idle':
                cur = None
    return {
        'pid': str(sel.get('participant')), 'session': sel.get('session'),
        'd': A.D_LABEL.get(str(sel.get('skip_budget')).replace('None', 'inf'), str(sel.get('skip_budget'))),
        'instruction': acc.get('task'), 'end_t': end_t, 'game_end': round(game_end, 1),
        'window': round(window, 1),
        'instr_s': round(tot['instr'], 1), 'other_s': round(tot['other'], 1), 'idle_s': round(tot['idle'], 1),
        'other_wait_s': round(tot['other_wait'], 1),
        'other_outside_s': round(tot['other'] - tot['other_wait'], 1),
        'cook_t': cook_t, 'kind': 'chop' if end_kind == 'chop' else kind,
        'instr_all_s': round(tot_all['instr'], 1), 'other_all_s': round(tot_all['other'], 1),
        'idle_all_s': round(tot_all['idle'], 1),
        'episodes': [(round(a, 1), round(b, 1), sorted(h)) for a, b, h in episodes if b - a >= 0.39],
        'frames': (frames, labels),
    }


def find_replays(pids):
    out = {}
    for path in sorted(glob.glob(str(ROOT / 'results/experience1/replays/*exp_ring*.rep'))):
        if path.rsplit('-', 1)[-1][:8] < '20261007':
            continue
        try:
            sel = R.load(path).get('web_selection') or {}
        except Exception:
            continue
        pid = str(sel.get('participant') or '')
        if pid in pids and sel.get('session'):
            out[(pid, int(sel['session']))] = path    # 同じ回が 2 つあれば後のもの
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', default=','.join(A.VALID))
    ap.add_argument('--show', default='')
    args = ap.parse_args()
    pids = set(p.strip() for p in args.pids.split(',') if p.strip())
    reps = find_replays(pids)
    if args.show:
        pid, ses = args.show.split(':')
        r = analyze(reps[(pid, int(ses))])
        frames, labels = r['frames']
        prev = None
        for (t0, dt, idle, hold, evs, _a), lab in zip(frames, labels):
            key = (lab, hold)
            if key != prev or evs:
                print('%5.1f %-6s %-34s %s' % (t0, lab, hold or '-', ' '.join(evs)))
                prev = key
        print({k: v for k, v in r.items() if k != 'frames'})
        return
    rows = []
    for (pid, ses), path in sorted(reps.items()):
        r = analyze(path)
        r.pop('frames')
        rows.append(r)
        print(r['pid'], r['session'], r['d'], r['instruction'], '終わり', r['end_t'], '煮始め', r['cook_t'],
              '指示', r['instr_s'], 'ほか(待ちの外)', r['other_outside_s'], 'ほか(煮える間)', r['other_wait_s'],
              '止まる', r['idle_s'], flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = A.OUT / 'ai_time_use.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('出力:', out)


if __name__ == '__main__':
    main()
