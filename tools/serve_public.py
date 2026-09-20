"""低遅延の入口(Cloudflare のトンネル)を立てて、固定 URL の行き先を更新する。

固定 URL(Tailscale Funnel)は中継が遠く、同じ Wi-Fi からでも往復1秒近く
かかることがあった。Cloudflare のトンネルは往復 30ms 程度で通るので、
固定 URL は「受付」だけにして、遊ぶ画面はこちらへ送る。

    python server.py --host 0.0.0.0      # 先にゲームのサーバーを起動しておく
    python tools/serve_public.py         # 別の窓で。URL を表示して待つ

この道具が立てる URL は起動のたびに変わるが、その URL を
`.cache/public_url.txt` に書くので、固定 URL を開いた人は自動でそちらへ
飛ぶ。止めるとファイルを消すので、固定 URL はそのまま遊べる画面に戻る。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL_FILE = ROOT / '.cache' / 'public_url.txt'
URL_RE = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')

CANDIDATES = [
    'cloudflared',
    r'C:\Program Files (x86)\cloudflared\cloudflared.exe',
    r'C:\Program Files\cloudflared\cloudflared.exe',
]


def find_cloudflared():
    for c in CANDIDATES:
        path = shutil.which(c) if os.sep not in c else (c if Path(c).exists() else None)
        if path:
            return path
    return None


def wait_for_server(port, seconds=30):
    for _ in range(seconds * 2):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/api/slot', timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000, help='ゲームのサーバーのポート')
    ap.add_argument('--url', default=None,
                    help='すでに立っているトンネルの URL を使う(立てずに書くだけ)')
    args = ap.parse_args()

    URL_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.url:
        URL_FILE.write_text(args.url, encoding='utf-8')
        print(f'行き先を書きました: {args.url}')
        return

    exe = find_cloudflared()
    if not exe:
        print('cloudflared が見つかりません。次で入れてください:')
        print('  winget install --id Cloudflare.cloudflared -e')
        return 1

    if not wait_for_server(args.port, 3):
        print(f'!! ポート {args.port} でゲームのサーバーが応答しません。'
              '先に python server.py --host 0.0.0.0 を起動してください。')

    proc = subprocess.Popen(
        [exe, 'tunnel', '--url', f'http://localhost:{args.port}', '--no-autoupdate'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1)

    found = threading.Event()

    def pump():
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m and not found.is_set():
                url = m.group(0)
                URL_FILE.write_text(url, encoding='utf-8')
                found.set()
                print('=' * 60)
                print('  遊ぶ URL(低遅延):', url)
                print('  受付の固定 URL  : https://desktop-1.tail9a3ca5.ts.net/')
                print('    -> 固定 URL を開いた人は、上の URL へ自動で移動します')
                print('  家の中から       : http://192.168.3.7:%d/' % args.port)
                print('=' * 60, flush=True)
            elif 'ERR' in line or 'error' in line.lower():
                print(line.rstrip(), flush=True)

    threading.Thread(target=pump, daemon=True).start()

    try:
        while proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        # 止めたら行き先を消す。固定 URL はそのまま遊べる画面に戻る。
        try:
            URL_FILE.unlink()
        except OSError:
            pass
        print('トンネルを止めました(固定 URL は元の画面に戻ります)')


if __name__ == '__main__':
    sys.exit(main() or 0)
