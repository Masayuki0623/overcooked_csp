"""実験の記録がそろって保存されているかを確かめる。

    python tools/check_records.py            # 全員(動作確認の参加者も)
    python tools/check_records.py --real     # 動作確認(テスト実行か=1)の参加者を除く
    python tools/check_records.py p15        # 1人だけ

参加者ごとに、各ファイルの行数を並べ、足りない・食い違うところに印を付ける。
記録のファイルは読むだけで、書き換えない。Excel で開いていても動く。

見るところ(パターン7 = 本番):
  ゲーム       web_sessions.csv の「正式な回か」=1 の行
  アンケート   exp_qualitative.csv
  統合         all_in_one_pattern7.csv(と全パターン混在の all_in_one.csv)
  指示の文     web_instruction_texts.csv の「確定」の行(ゲームごとに 1 つ)
  進み具合     assignments.json の done と、名簿の「完了したか」
  退避中の行   ○○-pending.csv(Excel で開いていて書けなかった行。閉じれば戻る)
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results' / 'experience1'
PATTERN = '7'
GAMES = 4


def read_rows(name):
    """本体と退避ファイル(-pending)の行。見出しの下の説明行は飛ばす。"""
    out = []
    p = DIR / name
    for q in (p, p.with_name(f'{p.stem}-pending{p.suffix}')):
        if not q.exists():
            continue
        try:
            with q.open('r', encoding='utf-8-sig', newline='') as f:
                out += [r for r in csv.DictReader(f) if str(r.get('記録時刻', '')).startswith('20')
                        or str(r.get('同意した日時', '')).startswith('20')]
        except OSError as e:
            print(f'  !! {q.name} を読めません: {e}')
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    real_only = '--real' in sys.argv
    roster = read_rows('participant_roster.csv')
    sessions = [r for r in read_rows('web_sessions.csv') if str(r.get('パターン')) == PATTERN]
    qual = [r for r in read_rows('exp_qualitative.csv') if str(r.get('パターン')) == PATTERN]
    allp = read_rows(f'all_in_one_pattern{PATTERN}.csv')
    texts = [r for r in read_rows('web_instruction_texts.csv') if str(r.get('パターン')) == PATTERN]
    try:
        assign = json.loads((DIR / 'assignments.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        assign = {}

    pending = sorted(p.name for p in DIR.glob('*-pending.csv'))
    print(f'記録の場所: {DIR}')
    print('退避中のファイル(閉じれば自動で戻る):', ', '.join(pending) if pending else 'なし')
    print()

    def by_pid(rows, key='参加者ID'):
        d = defaultdict(list)
        for r in rows:
            d[str(r.get(key, '')).strip()].append(r)
        return d

    s_by, q_by, a_by, t_by = by_pid(sessions), by_pid(qual), by_pid(allp), by_pid(texts)
    pids = [str(r.get('参加者番号', '')).strip() for r in roster]
    pids = [p for p in pids if p and (not args or p in args)]
    shown = 0
    for pid in pids:
        info = next(r for r in roster if str(r.get('参加者番号', '')).strip() == pid)
        test = str(info.get('テスト実行か', '')).strip() == '1'
        if real_only and test:
            continue
        rec = next((v for k, v in assign.items() if k.startswith(f'{pid}#p{PATTERN}@')), None)
        if rec is None and not s_by.get(pid) and not q_by.get(pid):
            continue                     # このパターンを遊んでいない人
        shown += 1
        games = [r for r in s_by.get(pid, []) if str(r.get('正式な回か')) == '1']
        g_sess = Counter(str(r.get('セッション番号')) for r in games)
        q_sess = Counter(str(r.get('セッション番号')) for r in q_by.get(pid, []))
        a_sess = Counter(str(r.get('セッション番号')) for r in a_by.get(pid, []))
        confirms = Counter(str(r.get('セッション番号')) for r in t_by.get(pid, []) if r.get('種類') == 'confirm')
        done = int((rec or {}).get('done', 0) or 0)
        problems = []
        for s in sorted(set(g_sess) | set(q_sess) | set(a_sess), key=lambda x: int(x) if x.isdigit() else 99):
            if g_sess.get(s) and not q_sess.get(s):
                problems.append(f'セッション{s}: ゲームはあるがアンケートが無い(まだ答えていない?)')
            if q_sess.get(s) and not a_sess.get(s):
                problems.append(f'セッション{s}: アンケートはあるが統合ファイルに行が無い')
            if g_sess.get(s) and not confirms.get(s):
                problems.append(f'セッション{s}: 確定した指示の文が無い')
            if a_sess.get(s, 0) > 1:
                problems.append(f'セッション{s}: 統合ファイルに {a_sess[s]} 行(やり直し)')
        for r in a_by.get(pid, []):
            if str(r.get('完了したか')) == '1' and not r.get('スコア_かかった時間_秒'):
                problems.append(f"セッション{r.get('セッション番号')}: 出し切ったのにスコアが空")
        if done >= GAMES and str(info.get('完了したか', '')).strip() != '1':
            problems.append('4 回とも終えたのに、名簿の「完了したか」が 1 になっていない')
        print(f"{pid}{' [テスト]' if test else ''}  名簿の完了={info.get('完了したか', '')}  "
              f"進み={done}/{GAMES}  ゲーム={len(games)}  アンケート={sum(q_sess.values())}  "
              f"統合={sum(a_sess.values())}  指示の確定={sum(confirms.values())}  "
              f"送った文={sum(1 for r in t_by.get(pid, []) if r.get('種類') == 'send')}")
        for p in problems:
            print('   !!', p)
        if not problems:
            print('   OK')
    if not shown:
        print('(該当する参加者はいません)')


if __name__ == '__main__':
    main()
