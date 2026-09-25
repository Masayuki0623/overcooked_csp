# -*- coding: utf-8 -*-
"""各回のリプレイを再生して、スープ(鍋)の時刻を取り出す。

  t_in    鍋に最初の材料が入った時刻
  t_cook  煮込みが始まった時刻(鍋がそろった)
  t_done  煮上がった時刻
  t_serve 提供した時刻
"""
import csv, glob, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils.replay import Replay

REPLAY_DIR = ROOT / 'agent' / 'agent' / 'replay'


def rebuild(sel):
    name = sel['map']
    recipes = sel.get('recipes') or []
    if (not name.startswith('tutorial_')
            and not any(f in r for r in recipes
                        for f in ('Apple', 'Orange', 'Banana', 'Juice'))):
        name = f"{name}_veg"
    kw = dict(MAP_SETTINGS[name])
    kw['order_recipes'] = tuple(recipes)
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def pot_timeline(path):
    rep = Replay.from_file(path)
    his = list(rep)
    try:
        sel = rep['web_selection']
    except Exception:
        sel = None
    if not sel:
        return None
    env = rebuild(sel)
    steps = [h for h in his if h['name'] == 'env.step']
    pots = [g for g in env.world.get_object_list()
            if getattr(g, 'name', '') == 'Pot']
    pot_locs = {g.location for g in env.world.objects.get('Pot', [])} if hasattr(env.world, 'objects') else set()
    out = {'t_in': None, 't_cook': None, 't_done': None}
    for h in steps:
        acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0))
                for a in env.sim_agents}
        env.step(acts, passed_time=h['args'].get('passed_time', 0.2))
        t = env.current_time
        names = []
        for key, objs in env.world.objects.items():
            for o in objs:
                fn = getattr(o, 'full_name', None)
                if fn:
                    names.append((fn, getattr(o, 'location', None)))
        for fn, loc in names:
            if out['t_in'] is None and loc in pot_locs and fn.startswith('Chopped'):
                out['t_in'] = t
            if out['t_cook'] is None and 'Cooking' in fn:
                out['t_cook'] = t
                if out['t_in'] is None:
                    out['t_in'] = t
            if out['t_done'] is None and 'Cooked' in fn:
                out['t_done'] = t
    log = getattr(env, 'delivery_log', None)
    out['deliveries'] = [dict(d) for d in (log or [])]
    out['pot_locs'] = sorted(pot_locs)
    out['map'] = sel.get('map')
    out['recipes'] = sel.get('recipes')
    out['skip_budget'] = sel.get('skip_budget')
    return out


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'results/web_sessions.csv', encoding='utf-8'))]
    want = [r for r in rows
            if (r['participant_id'].startswith('924test') or r['participant_id'].startswith('925'))
            and r['accepted'] == '1']
    files = {}
    for f in glob.glob(str(REPLAY_DIR / 'web*.rep')):
        m = re.search(r'(\d{8}_\d{6})\.rep$', f)
        if m:
            files[m.group(1)] = f
    res = []
    for r in want:
        ts = r['timestamp']
        key = re.sub(r'[-:T]', '', ts)[:15]
        key = key[:8] + '_' + key[8:14]
        f = files.get(key)
        if not f:
            print('リプレイなし', r['participant_id'], ts, file=sys.stderr)
            continue
        try:
            tl = pot_timeline(f)
        except Exception as e:
            print('失敗', ts, type(e).__name__, e, file=sys.stderr)
            continue
        if not tl:
            continue
        tl['pid'] = r['participant_id']
        tl['ts'] = ts
        tl['makespan'] = float(r['makespan_s'] or 0)
        tl['instruction'] = r['instruction']
        tl['serve_times'] = r['serve_times_s']
        tl['serve_dishes'] = r['serve_dishes']
        tl['loss'] = r['loss_seconds']
        tl['natural_rank'] = r['natural_rank']
        tl['exec_rank'] = r['exec_rank']
        tl['start_gain'] = r['start_gain_s']
        res.append(tl)
        print('.', end='', flush=True, file=sys.stderr)
    out = ROOT / 'results' / 'soup_timeline.json'
    json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n書き出し', out, len(res), file=sys.stderr)


main()
