import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, MutableMapping

from xtuner.v1.utils import get_logger


_INITIALIZED = False
_INIT_FAILED = False


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def enabled() -> bool:
    exporter = (os.environ.get("OTEL_TRACES_EXPORTER") or "").strip().lower()
    if exporter in {"none", "false", "off", "0"}:
        return False
    return (
        _truthy(os.environ.get("XTUNER_OTEL_ENABLED"))
        or _truthy(os.environ.get("AGENT_OTEL_ENABLED"))
        or bool(
            os.environ.get("OTEL_TRACES_EXPORTER")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        )
    )


def _init_otel() -> None:
    global _INITIALIZED, _INIT_FAILED
    if _INITIALIZED or _INIT_FAILED or not enabled():
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        service_name = os.environ.get("XTUNER_OTEL_SERVICE_NAME") or os.environ.get(
            "OTEL_SERVICE_NAME", "xtuner-rollout"
        )
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.namespace": "xtuner",
                    "service.instance.id": f"{os.uname().nodename}-{os.getpid()}",
                    "process.pid": os.getpid(),
                }
            )
        )

        traces_exporter = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").strip().lower()
        if traces_exporter == "console":
            exporter = ConsoleSpanExporter()
        elif traces_exporter in {"otlp", "otlp_proto_http", "otlp_proto_grpc"}:
            protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").strip().lower()
            if traces_exporter == "otlp_proto_grpc" or protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        else:
            get_logger().warning(f"[otel] unsupported OTEL_TRACES_EXPORTER={traces_exporter}; disabled")
            _INIT_FAILED = True
            return

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _INITIALIZED = True
        get_logger().info(f"[otel] enabled service.name={service_name}")
    except Exception as exc:
        _INIT_FAILED = True
        get_logger().warning(f"[otel] init failed: {type(exc).__name__}: {exc}")


def tracer():
    _init_otel()
    from opentelemetry import trace

    return trace.get_tracer("xtuner.v1.rl.rollout")


def extract_context(headers: Mapping[str, str] | None):
    if not enabled():
        return None
    try:
        from opentelemetry import propagate

        return propagate.extract(headers or {})
    except Exception:
        return None


def inject_context(headers: MutableMapping[str, str]) -> None:
    if not enabled():
        return
    try:
        from opentelemetry import propagate

        propagate.inject(headers)
    except Exception:
        return


@contextmanager
def use_context(context: Any) -> Iterator[None]:
    if not enabled() or context is None:
        yield
        return

    token = None
    try:
        from opentelemetry import context as otel_context

        token = otel_context.attach(context)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            otel_context.detach(token)
        except Exception:
            return


def set_attrs(span: Any, **attrs: Any) -> None:
    if span is None:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        try:
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
        except Exception:
            continue


def record_exception(span: Any, exc: BaseException) -> None:
    if span is None:
        return
    try:
        span.record_exception(exc)
        span.set_attribute("error.type", type(exc).__name__)
        span.set_attribute("error.message", str(exc))
    except Exception:
        return


def begin_span(name: str, context: Any = None, **attrs: Any) -> Any:
    if not enabled():
        return None
    try:
        span = tracer().start_span(name, context=context)
        set_attrs(span, **attrs)
        return span
    except Exception:
        return None


def end_span(span: Any, exc: BaseException | None = None, **attrs: Any) -> None:
    if span is None:
        return
    if exc is not None:
        record_exception(span, exc)
    set_attrs(span, **attrs)
    try:
        span.end()
    except Exception:
        return


@contextmanager
def start_span(name: str, context: Any = None, **attrs: Any) -> Iterator[Any]:
    if not enabled():
        yield None
        return

    try:
        with tracer().start_as_current_span(name, context=context) as span:
            set_attrs(span, **attrs)
            try:
                yield span
            except Exception as exc:
                record_exception(span, exc)
                raise
    except Exception:
        if _INIT_FAILED:
            yield None
            return
        raise
