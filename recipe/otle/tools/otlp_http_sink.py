#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)


def _hex(value: bytes) -> str:
    return value.hex()


def _attrs(attrs) -> dict:
    result = {}
    for attr in attrs:
        val = attr.value
        if val.HasField("string_value"):
            result[attr.key] = val.string_value
        elif val.HasField("bool_value"):
            result[attr.key] = val.bool_value
        elif val.HasField("int_value"):
            result[attr.key] = val.int_value
        elif val.HasField("double_value"):
            result[attr.key] = val.double_value
        else:
            result[attr.key] = MessageToDict(val)
    return result


def build_handler(root: Path):
    raw_dir = root / "received"
    summary_path = root / "spans.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        server_version = "otlp-http-sink/0.1"

        def do_POST(self) -> None:
            if self.path != "/v1/traces":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length)
            ts = time.time()
            raw_path = raw_dir / f"{ts:.6f}_{id(body)}.pb"
            raw_path.write_bytes(body)

            req = ExportTraceServiceRequest()
            try:
                req.ParseFromString(body)
                count = 0
                with summary_path.open("a", encoding="utf-8") as fp:
                    for resource_span in req.resource_spans:
                        resource = _attrs(resource_span.resource.attributes)
                        for scope_span in resource_span.scope_spans:
                            scope = {
                                "name": scope_span.scope.name,
                                "version": scope_span.scope.version,
                            }
                            for span in scope_span.spans:
                                count += 1
                                fp.write(
                                    json.dumps(
                                        {
                                            "recv_ts": ts,
                                            "trace_id": _hex(span.trace_id),
                                            "span_id": _hex(span.span_id),
                                            "parent_span_id": _hex(span.parent_span_id),
                                            "name": span.name,
                                            "start_unix_nano": span.start_time_unix_nano,
                                            "end_unix_nano": span.end_time_unix_nano,
                                            "duration_ms": (
                                                span.end_time_unix_nano - span.start_time_unix_nano
                                            )
                                            / 1e6,
                                            "status": span.status.code,
                                            "status_message": span.status.message,
                                            "attributes": _attrs(span.attributes),
                                            "resource": resource,
                                            "scope": scope,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                print(f"received traces bytes={len(body)} spans={count} raw={raw_path}", flush=True)
            except Exception as exc:
                print(f"failed to decode traces bytes={len(body)} raw={raw_path}: {exc}", flush=True)

            resp = ExportTraceServiceResponse().SerializeToString()
            self.send_response(200)
            self.send_header("content-type", "application/x-protobuf")
            self.send_header("content-length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--root", default="/tmp/otelcol")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(root))
    print(
        f"OTLP HTTP sink listening on {args.host}:{args.port}, summaries -> {root / 'spans.jsonl'}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
