"""古い results/web_sessions.csv を新しい列そろえに直し、L をリプレイから埋める。

記録する項目が増えると、server.py は古いファイルを別名にして残す
(列がずれて読めなくなるのを防ぐため)。この道具はその退避ファイルを拾って
1つの表に戻し、あとから増えた項目をリプレイから補う。

補えるのは、リプレイに残っている次の値。
    loss_seconds / baseline_seconds / constrained_seconds / loss_status /
    loss_num_tasks        <- instruction_time_loss (時間損失量 L(d) と内訳)
    instruction_accepted_s <- instruction_accepted (指示を受け取った時刻)
    wait_after_instruction_s は、着手時刻との引き算で出す。

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='実際に書き戻す')
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
    for r in rows:
        print('%-19s %-9s %-2s %-4s %-12s %6s %6s %6s %6s%s'
              % (r.get('timestamp', '')[:19], r.get('participant_id', ''),
                 r.get('session', ''), r.get('skip_budget', ''),
                 r.get('instruction', '') or '-',
                 r.get('wait_after_instruction_s') or r.get('wait_seconds') or '-',
                 r.get('loss_seconds', '') if r.get('loss_seconds', '') != '' else '-',
                 r.get('baseline_seconds', '') or '-',
                 r.get('makespan_s', '') or '-',
                 '  (打ち切り)' if r.get('aborted') == '1' else ''))

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
