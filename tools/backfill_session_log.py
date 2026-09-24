"""古い results/web_sessions.csv を新しい列そろえに直し、L をリプレイから埋める。

記録する項目が増えると、server.py は古いファイルを別名にして残す
(列がずれて読めなくなるのを防ぐため)。この道具はその退避ファイルを拾って
1つの表に戻し、あとから増えた項目をリプレイから補う。

補えるのは、リプレイに残っている次の値。
    loss_seconds / baseline_seconds / constrained_seconds / loss_status /
    loss_num_tasks        <- instruction_time_loss (時間損失量 L(d) と内訳)
    instruction_accepted_s <- instruction_accepted (指示を受け取った時刻)
    wait_after_instruction_s は、着手時刻との引き算で出す。
    serve_times_s / serve_dishes <- リプレイをその場で再生して、料理を
        出した時刻を拾う(記録に残っていないので作り直す。1件あたり数秒)。

    python tools/backfill_session_log.py            # 中身を見るだけ
    python tools/backfill_session_log.py --write    # 実際に書き戻す
"""
import argparse
import csv
import io
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.replay import Replay  # noqa: E402

LOG = ROOT / 'results' / 'web_sessions.csv'
REPLAY_DIR = ROOT / 'agent' / 'agent' / 'replay'
# リプレイはゲームが終わった瞬間、CSV はその少し後に書かれる。
# 同じ回とみなす時間差の上限。
MATCH_WINDOW_S = 60


def read_rows(path):
    if not path.exists():
        return []
    with io.open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def collect_rows():
    """本体と退避ファイルを集めて、同じ回を1つにまとめる。"""
    paths = [LOG] + sorted(LOG.parent.glob('web_sessions-*.csv'))
    seen = {}
    for path in paths:
        for row in read_rows(path):
            key = (row.get('participant_id'), row.get('session'),
                   row.get('timestamp'))
            cur = seen.setdefault(key, {})
            for k, v in row.items():
                # 埋まっている値を優先する(退避ファイル側にしかない列がある)
                if v not in (None, '') or k not in cur:
                    if v not in (None, '') or cur.get(k) in (None, ''):
                        cur[k] = v
    return [seen[k] for k in sorted(seen, key=lambda k: (k[2] or '', k[0] or ''))]


def replay_index():
    """リプレイを「終わった時刻」で引けるようにする。"""
    out = []
    for path in REPLAY_DIR.glob('web*.rep'):
        # web0-exp_ring-CSP-human-20260922_184115.rep
        stamp = path.stem.split('-')[-1]
        try:
            when = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
        except ValueError:
            continue
        out.append((when, path))
    return sorted(out)


def facts_from(path):
    """リプレイから、あとで増えた項目を取り出す。"""
    try:
        rep = Replay.from_file(path)
    except Exception:
        return {}
    loss, acc = None, None
    for h in rep:
        if h['name'] == 'instruction_time_loss':
            loss = h['args']
        elif h['name'] == 'instruction_accepted':
            acc = h['args']
    got = {}
    if loss:
        got['loss_seconds'] = loss.get('loss_seconds')
        got['baseline_seconds'] = loss.get('baseline_seconds')
        got['constrained_seconds'] = loss.get('constrained_seconds')
        got['loss_status'] = loss.get('status') or ''
        got['loss_num_tasks'] = loss.get('num_tasks')
    if acc and acc.get('accepted_time_env') is not None:
        got['instruction_accepted_s'] = round(float(acc['accepted_time_env']), 1)
    return {k: v for k, v in got.items() if v is not None}


def serve_times_from(path):
    """リプレイを再生して、1品ずつ出せた時刻を拾う。

    提供の時刻はどこにも書き出されていないので、同じ地図・同じ注文で
    盤面を作り直し、記録どおりの行動を流し込んで数え直す。
    """
    import replay_trace as RT
    try:
        info = RT.load(path)
        sel = info.get('web_selection')
        if not sel:
            return {}
        env = RT.rebuild(sel)
        for h in info['his']:
            if h['name'] != 'env.step':
                continue
            acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0))
                    for a in env.sim_agents}
            env.step(acts, passed_time=h['args'].get('passed_time', 0.2))
    except Exception as e:
        print(f'  ({Path(path).name} は再生できませんでした: {e})')
        return {}
    log = list(getattr(env, 'delivery_log', None) or [])
    ok = [d for d in log if d.get('ok', True)]
    if not log:
        return {}
    return {'serve_times_s': '|'.join(str(d['time']) for d in ok),
            'serve_dishes': '|'.join(d['dish'] for d in ok),
            'misserved': len(log) - len(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='実際に書き戻す')
    ap.add_argument('--replay-sim', action='store_true',
                    help='リプレイを再生して提供時刻も補う(時間がかかる)')
    args = ap.parse_args()

    # 列そろえは server.py の定義をそのまま使う(食い違いを起こさないため)。
    # SESSION_FIELDS は WebGamePlay の中にある。
    import server as S
    fields = list(S.WebGamePlay.SESSION_FIELDS)

    rows = collect_rows()
    reps = replay_index()
    filled = 0
    for row in rows:
        when = row.get('timestamp')
        if not when:
            continue
        try:
            t = datetime.fromisoformat(when)
        except ValueError:
            continue
        best = None
        for rep_when, path in reps:
            gap = abs((t - rep_when).total_seconds())
            if gap <= MATCH_WINDOW_S and (best is None or gap < best[0]):
                best = (gap, path)
        added = False
        if best is not None:
            for k, v in facts_from(best[1]).items():
                if k in fields and row.get(k) in (None, ''):
                    row[k] = v
                    added = True
            if (args.replay_sim and 'serve_times_s' in fields
                    and row.get('serve_times_s') in (None, '')):
                for k, v in serve_times_from(best[1]).items():
                    row[k] = v
                    added = True
        # 指示からの経過秒は、受け取った時刻との引き算で出せる
        if row.get('wait_after_instruction_s') in (None, ''):
            try:
                st = float(row.get('wait_seconds'))
                ac = float(row.get('instruction_accepted_s'))
                if row.get('wait_censored') != '1':
                    row['wait_after_instruction_s'] = round(st - ac, 1)
                    added = True
            except (TypeError, ValueError):
                pass
        filled += int(added)

    print(f'集めた行: {len(rows)} / リプレイから補えた行: {filled}')
    print()
    print('%-19s %-9s %-2s %-4s %-12s %6s %6s %6s %6s'
          % ('時刻', '参加者', '回', 'skip', '指示', '待ち秒', 'L(秒)', 'f(秒)', '所要'))
    # 提供の時刻は行ごとに長さが変わるので、表の右に添える。
    for r in rows:
        print('%-19s %-9s %-2s %-4s %-12s %6s %6s %6s %6s%s'
              % (r.get('timestamp', '')[:19], r.get('participant_id', ''),
                 r.get('session', ''), r.get('skip_budget', ''),
                 r.get('instruction', '') or '-',
                 r.get('wait_after_instruction_s') or r.get('wait_seconds') or '-',
                 r.get('loss_seconds', '') if r.get('loss_seconds', '') != '' else '-',
                 r.get('baseline_seconds', '') or '-',
                 r.get('makespan_s', '') or '-',
                 (('  提供 ' + r['serve_times_s']) if r.get('serve_times_s') else '')
                 + ('  (打ち切り)' if r.get('aborted') == '1' else '')))

    if not args.write:
        print()
        print('(--write を付けると results/web_sessions.csv へ書き戻します)')
        return 0

    with io.open(LOG, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print()
    print(f'{LOG} に {len(rows)} 行を書き戻しました')
    return 0


if __name__ == '__main__':
    sys.exit(main())
