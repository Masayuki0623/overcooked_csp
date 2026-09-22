"""URL 1つで、何人かが同時に遊べるようにする入口。

ゲームのサーバー(server.py)は pygame の画面をプロセスに1つしか持てない
ので、1プロセス = 1人までしか遊べない。ここでは人数ぶんの server.py を裏で
立てておき、表の URL に来た人を空いている席へ回す。参加者に配る URL は
1つのままでよく、cloudflared / Tailscale の設定も今までどおり。

    python server_multi.py                     # 4人ぶん(既定)
    python server_multi.py --players 2
    python server_multi.py --map exp_ring      # server.py の引数はそのまま渡る

席は端末ごとの目印(cookie)で覚える。遊んでいる途中で画面を読み込み直しても
同じ席に戻り、別の人の画面が出てくることはない。全部ふさがっているときは、
これまでと同じ「別の人がプレイ中」の画面が出る。

人数の上限は、このPC(8コア)で測った上で 4 にしてある。1人ぶんが CPU
0.17コア・メモリ 264MB なので、4人で 0.7コア・1.1GB。
"""
import argparse
import asyncio
import contextlib
import subprocess
import sys
import time
from pathlib import Path

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response

ROOT = Path(__file__).resolve().parent
# このPC(i7-11700 8コア/16GB)で測った上限。1人ぶんが CPU 0.17コア・
# メモリ 264MB。4人でも、1人のときとコマ落ちはほぼ変わらなかった。
MAX_PLAYERS = 4
SEAT_COOKIE = 'ocsp_seat'
# 席を渡してから、その人が実際につなぎに来るまでの取り置き時間。
# これが無いと、2人が同時に画面を開いたとき両方に同じ席を渡してしまう。
HOLD_SECONDS = 25.0
# 空き状況の問い合わせ結果をこれだけ使い回す(毎回4台に聞くと遅い)。
SLOT_CACHE_SECONDS = 0.5
# 受け渡しの途中で書き換えてはいけない見出し。
HOP_BY_HOP = {'connection', 'keep-alive', 'transfer-encoding', 'upgrade',
              'content-encoding', 'content-length', 'te', 'trailer',
              'proxy-authorization', 'proxy-authenticate'}
# 受付自身が付け直す見出し。そのまま通すと二重になる。
DROP_FROM_GAME = {'date', 'server'}


class Seats:
    """裏で動いている server.py を束ねて、空いている席を選ぶ。"""

    def __init__(self, ports):
        self.ports = list(ports)
        self._held_until = [0.0] * len(ports)
        self._cache = ([], 0.0)
        self._lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=1.0)

    async def close(self):
        await self.client.aclose()

    async def _slots(self):
        """各席の使用状況。少しの間だけ使い回す。"""
        rows, at = self._cache
        if rows and time.time() - at < SLOT_CACHE_SECONDS:
            return rows

        async def ask(port):
            try:
                r = await self.client.get(f'http://127.0.0.1:{port}/api/slot')
                return r.json()
            except Exception:
                # 落ちている / まだ起きていない席は、埋まっている扱いにする。
                return {'player_connected': True, 'state': 'down'}

        rows = await asyncio.gather(*(ask(p) for p in self.ports))
        self._cache = (rows, time.time())
        return rows

    async def status(self):
        rows = await self._slots()
        now = time.time()
        return [{'seat': i, 'port': self.ports[i],
                 'held': self._held_until[i] > now, **row}
                for i, row in enumerate(rows)]

    async def pick(self, current=None):
        """席を決める。いまの席が空いていればそのまま使う。

        current: その端末がすでに持っている席(cookie)。
        戻り値は席の番号。全部ふさがっていても、待ち画面を出すために
        どこかの席へ回す(いまの席があればそこ)。
        """
        async with self._lock:
            rows = await self._slots()
            now = time.time()

            def usable(i):
                return not rows[i].get('player_connected') and self._held_until[i] <= now

            if current is not None and 0 <= current < len(self.ports):
                if usable(current) or self._held_until[current] > now:
                    # 取り置き中の席は、その端末のために空けてある。
                    self._held_until[current] = now + HOLD_SECONDS
                    self._cache = ([], 0.0)
                    return current

            for i in range(len(self.ports)):
                if usable(i):
                    self._held_until[i] = now + HOLD_SECONDS
                    self._cache = ([], 0.0)
                    return i

            # 空きなし。いまの席、無ければ 0 番へ回して待ち画面を出す。
            return current if current is not None and 0 <= current < len(self.ports) else 0

    def note_connected(self, seat):
        """実際につながったので、取り置きは解いてよい(本人が占有する)。"""
        if 0 <= seat < len(self.ports):
            self._held_until[seat] = 0.0
            self._cache = ([], 0.0)


def seat_from_cookie(scope_cookies):
    try:
        return int(scope_cookies.get(SEAT_COOKIE, ''))
    except (TypeError, ValueError):
        return None


def build_app(seats: Seats):
    app = FastAPI(title='Overcooked CSP 受付')

    @app.get('/__seats')
    async def seat_status():
        """どの席が埋まっているかを見る(運用の確認用)。"""
        return JSONResponse({'seats': await seats.status()})

    @app.websocket('/ws')
    async def ws_proxy(client: WebSocket):
        seat = seat_from_cookie(client.cookies)
        if seat is None or not (0 <= seat < len(seats.ports)):
            seat = await seats.pick(None)
        seats.note_connected(seat)
        port = seats.ports[seat]

        await client.accept()
        # 誰がつないでいるかを裏のサーバーへ伝える。そのまま渡すと
        # 接続元がこの入口(127.0.0.1)になり、記録に残らない。
        fwd = {}
        ua = client.headers.get('user-agent')
        if ua:
            fwd['user-agent'] = ua
        real = client.client.host if client.client else None
        if real:
            fwd['x-forwarded-for'] = real
        for name in ('tailscale-funnel-request',):
            if name in client.headers:
                fwd[name] = client.headers[name]

        try:
            async with websockets.connect(f'ws://127.0.0.1:{port}/ws',
                                          additional_headers=fwd,
                                          max_size=None,
                                          ping_interval=None,
                                          open_timeout=10) as up:
                async def to_game():
                    while True:
                        msg = await client.receive()
                        if msg.get('type') == 'websocket.disconnect':
                            return
                        if msg.get('text') is not None:
                            await up.send(msg['text'])
                        elif msg.get('bytes') is not None:
                            await up.send(msg['bytes'])

                async def to_player():
                    async for msg in up:
                        if isinstance(msg, (bytes, bytearray)):
                            await client.send_bytes(bytes(msg))
                        else:
                            await client.send_text(msg)

                tasks = [asyncio.create_task(to_game()), asyncio.create_task(to_player())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
        except Exception as err:
            print(f'[multi] 席{seat} の中継が切れました: {err}')
        finally:
            with contextlib.suppress(Exception):
                await client.close()

    @app.api_route('/{path:path}',
                   methods=['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'])
    async def http_proxy(path: str, req: Request):
        current = seat_from_cookie(req.cookies)
        # 席を選び直すのは画面を開いたときだけ。その後の問い合わせは
        # 同じ席へ送らないと、別の人のゲームの状態が混ざる。
        if path == '' and req.method == 'GET':
            seat = await seats.pick(current)
        elif current is not None and 0 <= current < len(seats.ports):
            seat = current
        else:
            seat = await seats.pick(None)
        port = seats.ports[seat]

        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_BY_HOP}
        headers['x-forwarded-for'] = req.client.host if req.client else ''
        try:
            up = await seats.client.request(
                req.method, f'http://127.0.0.1:{port}/{path}',
                content=await req.body(), headers=headers,
                params=req.query_params, timeout=30.0)
        except Exception as err:
            return JSONResponse({'error': f'席{seat} につながりません: {err}'},
                                status_code=502)

        out = {k: v for k, v in up.headers.items()
               if k.lower() not in HOP_BY_HOP and k.lower() not in DROP_FROM_GAME}
        resp = Response(content=up.content, status_code=up.status_code,
                        headers=out, media_type=up.headers.get('content-type'))
        if seat != current:
            # 席は端末ごとに覚える。ブラウザを閉じるまで同じ席に戻る。
            resp.set_cookie(SEAT_COOKIE, str(seat), samesite='lax', path='/')
        return resp

    return app


def spawn_backends(players, base_port, passthrough):
    """人数ぶんの server.py を裏で立てる。"""
    procs = []
    for i in range(players):
        port = base_port + i
        cmd = [sys.executable, str(ROOT / 'server.py'),
               '--host', '127.0.0.1', '--port', str(port), '--instance', str(i),
               *passthrough]
        print(f'[multi] 席{i}: ポート {port} で起動します')
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    return procs


def wait_backends(ports, seconds=120):
    """全部の席が返事をするまで待つ。"""
    deadline = time.time() + seconds
    pending = list(ports)
    with httpx.Client(timeout=1.0) as c:
        while pending and time.time() < deadline:
            for port in list(pending):
                try:
                    c.get(f'http://127.0.0.1:{port}/api/slot')
                    pending.remove(port)
                    print(f'[multi] ポート {port} が応答しました')
                except Exception:
                    pass
            if pending:
                time.sleep(0.5)
    return not pending


def main():
    ap = argparse.ArgumentParser(
        description='URL 1つで複数人が同時に遊べるようにする入口',
        epilog='ここに書いていない引数は、そのまま server.py へ渡る')
    ap.add_argument('--players', type=int, default=4,
                    help='同時に遊べる人数(既定 4)')
    ap.add_argument('--host', default='0.0.0.0', help='受付が待ち受けるアドレス')
    ap.add_argument('--port', type=int, default=8000, help='受付のポート(参加者が開く)')
    ap.add_argument('--base-port', type=int, default=8001,
                    help='裏で立てるゲームのサーバーの最初のポート')
    args, passthrough = ap.parse_known_args()

    if args.players < 1:
        ap.error('--players は1以上にしてください')
    if args.players > MAX_PLAYERS:
        ap.error(f'--players は {MAX_PLAYERS} までです'
                 f'(このPCで測った上限。1人あたり CPU 0.17コア・メモリ 264MB)')
    if args.base_port <= args.port < args.base_port + args.players:
        ap.error('--port が裏のサーバーのポートと重なっています')

    ports = [args.base_port + i for i in range(args.players)]
    procs = spawn_backends(args.players, args.base_port, passthrough)

    try:
        if not wait_backends(ports):
            print('[multi] 起動しない席があります。そのまま受付を始めます')

        seats = Seats(ports)
        app = build_app(seats)
        shown = 'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
        print('=' * 60)
        print(f'  Overcooked CSP 受付 ({args.players}人まで同時に遊べます)')
        print(f'  URL: http://{shown}:{args.port}/')
        print(f'  席の状況: http://{shown}:{args.port}/__seats')
        print('=' * 60)
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
    except KeyboardInterrupt:
        print('\n[multi] 中断しました')
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate()
        for p in procs:
            with contextlib.suppress(Exception):
                p.wait(timeout=10)


if __name__ == '__main__':
    main()
