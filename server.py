"""
server.py
=========
零依赖后端：使用 Python 标准库 http.server 提供 REST API 并托管前端静态页面。

★ 多会话（房间）隔离

接口（REST / JSON）：
    GET  /api/state                       -> 当前会话状态（公开 + 真人合法私有信息）
    POST /api/new_game                    -> 重开本会话一局
    POST /api/team        {team:[pid...]} -> 真人作为队长提交队伍
    POST /api/speak       {text:str}      -> 真人在发言阶段发言（可在本回合内多次调用）
    POST /api/end_speak                   -> 真人结束本回合发言，让给下一位
    POST /api/end_discussion              -> 真人提前结束整段讨论，进入投票
    POST /api/vote        {approve:bool}  -> 真人投票
    POST /api/mission     {action:"success"|"fail"} -> 真人执行任务
    POST /api/assassinate {target:pid}    -> 真人（刺客）刺杀
"""

import json
import os
import threading
import time
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import llm
from game import Game

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
COOKIE = "avalon_sid"
MAX_GAMES = int(os.environ.get("AVALON_MAX_GAMES", "200"))
TTL = int(os.environ.get("AVALON_TTL", "1800"))


# ============================================================
# 会话：每个访客一局，自带后台推进 worker
# ============================================================
class Session:
    def __init__(self):
        self.game = Game()
        self.worker = None
        self.wguard = threading.Lock()
        self.last = time.time()
        self.worker_error = None

    def touch(self):
        self.last = time.time()

    def new_game(self, n_players=7):
        self.game = Game(n_players)
        self.worker_error = None

    def _advance_worker(self):
        try:
            self.game.advance()
        except Exception as exc:
            self.worker_error = f"AI 推进失败：{exc}"
            self.game._log(self.worker_error)

    def kick(self):
        """若没有正在跑的推进线程，则后台启动一个（推进 AI 至轮到真人/结束）。"""
        with self.wguard:
            if self.worker_error or (self.worker and self.worker.is_alive()):
                return
            self.worker = threading.Thread(target=self._advance_worker, daemon=True)
            self.worker.start()


_sessions = {}                      # sid -> Session
_slock = threading.Lock()


def _reap_locked():
    """回收空闲过久的会话；并把总数压到上限以内（控成本/内存）。"""
    now = time.time()
    for sid in [k for k, v in _sessions.items() if now - v.last > TTL]:
        _sessions.pop(sid, None)
    if len(_sessions) > MAX_GAMES:
        oldest = sorted(_sessions.items(), key=lambda kv: kv[1].last)
        for sid, _ in oldest[:len(_sessions) - MAX_GAMES]:
            _sessions.pop(sid, None)


def get_session(sid, create=True):
    """按 sid 取会话；不存在则（可选）新建。返回 (sid, Session, is_new)。

    sid 可以是外部指定的（URL 里带 ?sid=xxx）：这样脚本与浏览器能进同一局，
    用于"观战/自动演示"。不指定时仍按原逻辑生成随机 uuid。
    """
    with _slock:
        s = _sessions.get(sid) if sid else None
        if s is None and create:
            _reap_locked()
            if not sid:
                sid = uuid.uuid4().hex
            s = Session()
            _sessions[sid] = s
            s.touch()
            return sid, s, True
        if s:
            s.touch()
        return sid, s, False


class Handler(BaseHTTPRequestHandler):
    # --------- 工具 ---------
    def _sid_from_cookie(self):
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            return SimpleCookie(raw)[COOKIE].value
        except (KeyError, Exception):
            return None

    def _sid_from_query(self):
        """支持 ?sid=xxx：脚本与浏览器可用同一个 sid 进入同一局（观战/演示）。"""
        try:
            q = parse_qs(urlparse(self.path).query)
            return (q.get("sid") or [None])[0] or None
        except Exception:
            return None

    def _sid(self):
        return self._sid_from_query() or self._sid_from_cookie()

    def _send_json(self, obj, code=200, set_sid=None):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if set_sid:
            self.send_header("Set-Cookie",
                             f"{COOKIE}={set_sid}; Path=/; Max-Age={TTL}; SameSite=Lax")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _send_static(self, path, set_sid=None):
        rel = path.lstrip("/")
        if rel in ("", "index.html"):
            rel = "index.html"
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self.send_error(404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if set_sid:
            self.send_header("Set-Cookie",
                             f"{COOKIE}={set_sid}; Path=/; Max-Age={TTL}; SameSite=Lax")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

    # --------- 路由 ---------
    def do_GET(self):
        if self.path.startswith("/api/state"):
            sid, sess, is_new = get_session(self._sid(), create=True)
            sess.kick()                         # 自愈：需要 AI 行动则后台推进；GET 立即返回
            state = sess.game.view_for_human()
            if sess.worker_error:
                state["error"] = sess.worker_error
            self._send_json(state, set_sid=(sid if is_new else None))
            return
        q_sid = self._sid_from_query()
        if q_sid:
            # 让浏览器把 ?sid=xxx 记住：后续 /api/state 会走同一局
            get_session(q_sid, create=True)
        self._send_static(self.path.split("?")[0],
                          set_sid=(q_sid if q_sid and self.path.split("?")[0] in ("/", "/index.html", "") else None))

    def do_POST(self):
        body = self._read_body()
        path = self.path.split("?")[0]          # 允许 ?sid=xxx 之类的查询参数
        sid, sess, is_new = get_session(self._sid(), create=True)
        g = sess.game
        hid = g.human_id
        try:
            if path == "/api/new_game":
                n = body.get("n_players", 7)
                try:
                    n = max(5, min(10, int(n)))
                except (TypeError, ValueError):
                    n = 7
                sess.new_game(n)
                g = sess.game
            elif path == "/api/team":
                g.submit_team(hid, [int(x) for x in body.get("team", [])])
            elif path == "/api/speak":
                g.submit_speech(hid, body.get("text", ""))
            elif path == "/api/end_speak":
                g.end_speech(hid)
            elif path == "/api/end_discussion":
                g.end_discussion(hid)
            elif path == "/api/vote":
                g.submit_vote(hid, bool(body.get("approve")))
            elif path == "/api/mission":
                g.submit_mission(hid, body.get("action"))
            elif path == "/api/assassinate":
                g.do_assassinate(hid, int(body.get("target")))
            else:
                self.send_error(404)
                return
        except ValueError as e:
            self._send_json({"error": str(e)}, code=400, set_sid=(sid if is_new else None))
            return
        sess.kick()                             # 真人动作已记录，后台推进 AI
        state = g.view_for_human()
        if sess.worker_error:
            state["error"] = sess.worker_error
        self._send_json(state, set_sid=(sid if is_new else None))


def main():
    port = int(os.environ.get("PORT", "8000"))
    if not llm.is_available():
        raise RuntimeError(
            "未配置 LLM，项目只支持 LLM 模式。请在 config.py 中设置 LLM_API_KEY，"
            "或设置 LLM_API_KEY 环境变量后重启。"
        )
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"🤖 LLM 智能体：已启用（{llm.provider_name()} · {llm.model_name()}）")
    print(f"🎲 阿瓦隆已启动： http://0.0.0.0:{port}  （多会话：每位访客独立一局）")
    print(f"   上限 {MAX_GAMES} 局 / 空闲 {TTL}s 回收。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
