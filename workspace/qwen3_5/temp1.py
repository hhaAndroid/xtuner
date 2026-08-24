# replay_request.py
import ast
import json
from pathlib import Path

import httpx


def parse_temp_py_line(line: str):
    """
    temp.py 第 1 行形如：
    {payload_dict} http://x.x.x.x:port/generate {headers_dict} Server error '...'

    我们只解析前 3 段：payload / url / headers
    """
    line = line.strip()

    # 1) payload: 从行首开始的一个 dict 字面量
    if not line.startswith("{"):
        raise ValueError("temp.py line does not start with '{' payload dict")

    # 用大括号计数找到 payload dict 的结束位置
    depth = 0
    payload_end = None
    for i, ch in enumerate(line):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                payload_end = i + 1
                break
    if payload_end is None:
        raise ValueError("cannot find end of payload dict")

    payload_str = line[:payload_end]
    rest = line[payload_end:].lstrip()

    # 2) url: rest 的第一个 token
    parts = rest.split(None, 1)
    if len(parts) < 2:
        raise ValueError("cannot parse url after payload")
    url = parts[0]
    rest = parts[1].lstrip()

    # 3) headers: rest 开头的一个 dict 字面量
    if not rest.startswith("{"):
        raise ValueError("cannot find headers dict after url")

    depth = 0
    headers_end = None
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                headers_end = i + 1
                break
    if headers_end is None:
        raise ValueError("cannot find end of headers dict")

    headers_str = rest[:headers_end]

    payload = ast.literal_eval(payload_str)
    headers = ast.literal_eval(headers_str)

    return payload, url, headers


def main():
    temp_path = Path("temp.py")
    line = temp_path.read_text(encoding="utf-8").splitlines()[0]

    payload, url, headers = parse_temp_py_line(line)

    # 可选：避免传 "Bearer None"
    auth = headers.get("Authorization")
    if isinstance(auth, str) and auth.strip() in ("Bearer None", "None", ""):
        headers.pop("Authorization", None)

    # 如果你需要复现某个随机采样导致的问题，可以在 payload 里加 seed（前提是服务端支持）
    # payload.setdefault("seed", 1234)

    print("POST", url)
    print("headers =", headers)
    print("payload keys =", sorted(payload.keys()))

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        r = client.post(url, headers=headers, json=payload)

    print("status =", r.status_code)
    print("response headers =", dict(r.headers))

    # 尽量打印可读内容
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            data = r.json()
            print("json =", json.dumps(data, ensure_ascii=False, indent=2)[:20000])
        except Exception:
            print("text =", r.text[:20000])
    else:
        print("text =", r.text[:20000])


if __name__ == "__main__":
    main()