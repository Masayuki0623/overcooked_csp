"""誰かが遊びに来たら知らせる(サーバーとは別に動かす。サーバーの再起動は要らない)。

    python tools/notify_players.py            # 見張りを始める(Ctrl+C で止める)
    python tools/notify_players.py --test     # 通知のテストだけ

数秒ごとに、サーバーの席の状況(http://localhost:8000/__seats)とログ
(results/server.log)を読む。読むだけで、サーバーにもログにも書き込まない。

知らせる場面:
  - 誰かがつながった(どの席も空いていたところへ)
  - 新しく登録した(同意を受け取った。参加者番号つき)
  - ゲームが始まった(練習か本番か、地図)
知らせ方: Windows の通知(画面右下)と短い音。ゲームを全画面で遊んでいる間は
Windows が通知を隠すことがある(集中モード)ので、音も鳴らす。
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'results' / 'server.log'
SEATS_URL = 'http://localhost:8000/__seats'
POLL_S = 3
QUIET_S = 120          # 同じ席の「つながった」は、この秒数のあいだ繰り返し知らせない


def beep():
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except Exception:
        pass


def toast(title, body):
    """Windows の通知。だめならコンソールに出すだけ。"""
    print(time.strftime('%H:%M:%S'), title, '-', body, flush=True)
    beep()
    t = title.replace("'", "''")
    b = body.replace("'", "''")
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null;"
        f"$xml = '<toast><visual><binding template=\"ToastGeneric\"><text>{t}</text><text>{b}</text></binding></visual>"
        "<audio src=\"ms-winsoundevent:Notification.Default\"/></toast>';"
        "$doc = New-Object Windows.Data.Xml.Dom.XmlDocument; $doc.LoadXml($xml);"
        "$n = New-Object Windows.UI.Notifications.ToastNotification $doc;"
        "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show($n)"
    )
    try:
        subprocess.run(['powershell', '-NoProfile', '-Command', ps], timeout=15,
                       capture_output=True)
    except Exception as e:
        print('  (通知を出せませんでした:', e, ')', flush=True)


def read_seats():
    try:
        with urllib.request.urlopen(SEATS_URL, timeout=3) as r:
            return json.loads(r.read().decode('utf-8')).get('seats') or []
    except Exception:
        return None


def decode(line):
    """ログは席ごとのサーバーが Shift_JIS で書く行と UTF-8 の行が混ざる。"""
    for enc in ('utf-8', 'cp932'):
        try:
            return line.decode(enc)
        except UnicodeDecodeError:
            continue
    return line.decode('utf-8', errors='replace')


def follow_log(pos):
    """前に読んだところから先の行。"""
    try:
        size = LOG.stat().st_size
    except OSError:
        return pos, []
    if size < pos:
        pos = 0                      # ログが作り直された
    if size == pos:
        return pos, []
    with LOG.open('rb') as f:
        f.seek(pos)
        data = f.read()
    lines = data.split(b'\n')
    if not data.endswith(b'\n'):
        tail = lines.pop()           # 書きかけの行は次に回す
        pos += len(data) - len(tail)
    else:
        pos += len(data)
    return pos, [decode(x).rstrip('\r') for x in lines if x.strip()]


RE_CONSENT = re.compile(r'同意を受け取りました: (\S+)')
RE_SELECT = re.compile(r'#(\d+) 選択: (\S+) / (\S+)')
RE_START = re.compile(r'#(\d+) ゲームを開始します')


def main():
    if '--test' in sys.argv:
        toast('Overcooked 実験', '通知のテストです')
        return
    print('見張りを始めます(Ctrl+C で止める)', flush=True)
    toast('Overcooked 実験', '見張りを始めました。誰かが遊び始めたら知らせます')
    pos = LOG.stat().st_size if LOG.exists() else 0     # 今より前のログは見ない
    last_conn = {}
    prev = None
    last_select = {}
    while True:
        seats = read_seats()
        if seats is not None:
            now = time.time()
            for s in seats:
                i = s.get('seat')
                was = bool(prev and next((p for p in prev if p.get('seat') == i), {}).get('player_connected'))
                if s.get('player_connected') and not was and now - last_conn.get(i, 0) > QUIET_S:
                    last_conn[i] = now
                    toast('誰かがつながりました', f'席 {i}(全 {len(seats)} 席)')
            prev = seats
        pos, lines = follow_log(pos)
        for line in lines:
            m = RE_CONSENT.search(line)
            if m:
                toast('新しく登録しました', f'参加者 {m.group(1)}')
                continue
            m = RE_SELECT.search(line)
            if m:
                last_select[m.group(1)] = (m.group(2), m.group(3))
                continue
            m = RE_START.search(line)
            if m:
                game_map, preset = last_select.get(m.group(1), ('?', '?'))
                kind = '練習' if game_map.startswith('tutorial') or preset in ('None', 'experiment3') else '本番'
                toast('ゲームが始まりました', f'{kind}({game_map})')
        time.sleep(POLL_S)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('見張りを止めました')
