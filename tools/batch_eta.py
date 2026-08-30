"""並列で回しているバッチの進捗から、残り時間と完了予定時刻を出す。

    python tools/batch_eta.py "<ログのglob>"

進捗行「 n/m 件」を読み、ログの作成時刻からの経過で1件あたりの実測時間を
出す。シャードごとに終わる時刻が違うので、全体の完了は最も遅いシャードで
決まる。1/8 ほど進めば十分な精度になる。
"""
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta

COUNT = re.compile(r'(\d+)/(\d+) 件')


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'st_*.log'
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f'ログが見つかりません: {pattern}')
        return

    now = time.time()
    done = total = 0
    slowest = 0.0
    rows = []
    for path in paths:
        last = None
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = COUNT.search(line)
                    if m:
                        last = m
        except OSError:
            continue
        name = os.path.basename(path)
        started = _started(path)
        if last is None:
            rows.append(f'  {name}: まだ1件も終わっていません')
            continue
        n, m_total = int(last.group(1)), int(last.group(2))
        done += n
        total += m_total
        elapsed = max(1.0, now - started)
        per = elapsed / max(1, n)
        remain = per * (m_total - n)
        slowest = max(slowest, remain)
        rows.append(f'  {name}: {n}/{m_total} 件  1件{per:.0f}秒  残り約{remain / 60:.1f}分')

    print(chr(10).join(rows))

    if total:
        pct = 100 * done // total
        eta = datetime.now() + timedelta(seconds=slowest)
        print(f'  --- 合計 {done}/{total} 件 ({pct}%) ---')
        print(f'  完了予定 {eta.strftime("%H:%M")}（最も遅いシャードで約{slowest / 60:.1f}分）')


def _started(path):
    """そのシャードが動き始めた時刻(ログの作成時刻で代用)。"""
    try:
        st = os.stat(path)
        return getattr(st, 'st_ctime', st.st_mtime)
    except OSError:
        return time.time()


if __name__ == '__main__':
    main()
