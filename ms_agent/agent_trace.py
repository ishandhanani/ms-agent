# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal Dynamo agent trace integration helpers.

ms-agent only attaches Dynamo context to LLM requests and optionally publishes
tool lifecycle events. Dynamo owns normalized LLM tracing and trace sinks.
"""

import contextlib
import contextvars
import logging
import os
import struct
import threading
import time
import uuid
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

_CONTEXT: contextvars.ContextVar[Optional[Dict[str, str]]] = (
    contextvars.ContextVar('dynamo_agent_context', default=None))
_WORKFLOW_ID = os.environ.get('DYNAMO_AGENT_WORKFLOW_ID',
                              f'ms-agent-{uuid.uuid4().hex[:12]}')
_TOOL_EVENT_PUBLISHER: Optional[Any] = None
_TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED = False
_TOOL_EVENT_PUBLISHER_LOCK = threading.Lock()

_TOOL_EVENTS_ENDPOINT_ENVS = (
    'DYNAMO_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT',
    # Backward-compatible alias used by early local E2E wrappers.
    'DYNAMO_AGENT_TRACE_TOOL_ZMQ_ENDPOINT',
    # Accept Dynamo's server-side name when both processes share one env file.
    'DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT',
)
_TOOL_EVENTS_TOPIC_ENVS = (
    'DYNAMO_AGENT_TOOL_EVENTS_ZMQ_TOPIC',
    'DYNAMO_AGENT_TRACE_TOOL_ZMQ_TOPIC',
    'DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC',
)
_DEFAULT_ZMQ_STARTUP_DELAY_SECONDS = 0.5


def _first_env(names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class _ZmqToolEventPublisher:

    def __init__(self,
                 endpoint: str,
                 topic: str = '',
                 startup_delay_seconds: float = _DEFAULT_ZMQ_STARTUP_DELAY_SECONDS):
        import msgpack
        import zmq

        self._msgpack = msgpack
        self._topic = topic.encode('utf-8')
        self._seq = 0
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.bind(endpoint)
        if startup_delay_seconds > 0:
            time.sleep(startup_delay_seconds)

    def publish(self, record: Dict[str, Any]) -> None:
        self._seq += 1
        payload = self._msgpack.packb(record, use_bin_type=True)
        self._socket.send_multipart(
            [self._topic, struct.pack('>Q', self._seq), payload])


def configure_tool_event_publisher(publisher: Optional[Any]) -> None:
    """Register a best-effort publisher for Dynamo tool lifecycle events."""
    global _TOOL_EVENT_PUBLISHER, _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED
    with _TOOL_EVENT_PUBLISHER_LOCK:
        _TOOL_EVENT_PUBLISHER = publisher
        _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED = True


def init_tool_event_publisher_from_env() -> bool:
    """Initialize Dynamo tool-event publishing from environment variables."""
    global _TOOL_EVENT_PUBLISHER, _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED

    with _TOOL_EVENT_PUBLISHER_LOCK:
        if _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED:
            return _TOOL_EVENT_PUBLISHER is not None
        _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED = True

        endpoint = _first_env(_TOOL_EVENTS_ENDPOINT_ENVS)
        if not endpoint:
            return False
        topic = _first_env(_TOOL_EVENTS_TOPIC_ENVS) or ''

        try:
            _TOOL_EVENT_PUBLISHER = _ZmqToolEventPublisher(endpoint, topic)
        except Exception as exc:  # noqa
            logger.warning(
                'Dynamo tool-event publisher disabled: failed to bind %s: %s',
                endpoint, exc)
            _TOOL_EVENT_PUBLISHER = None
            return False

        logger.info('Dynamo tool-event publisher started on %s', endpoint)
        return True


def build_agent_context(
    agent_tag: str,
    workflow_type_id: Optional[str] = None,
    parent_program_id: Optional[str] = None,
) -> Dict[str, str]:
    workflow_type_id = (workflow_type_id
                        or os.environ.get('DYNAMO_AGENT_WORKFLOW_TYPE_ID')
                        or 'ms_agent')
    program_id = f'{_WORKFLOW_ID}:{agent_tag}:{uuid.uuid4().hex[:8]}'
    context = {
        'workflow_id': _WORKFLOW_ID,
        'workflow_type_id': workflow_type_id,
        'program_id': program_id,
    }
    if parent_program_id:
        context['parent_program_id'] = parent_program_id
    return context


@contextlib.contextmanager
def activate_context(agent_context: Optional[Dict[str, str]]) -> Iterator[None]:
    token = _CONTEXT.set(dict(agent_context) if agent_context else None)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_program_id() -> Optional[str]:
    context = _CONTEXT.get()
    if not context:
        return None
    return context.get('program_id')


def instrument_llm_request(
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach Dynamo context and x-request-id to an OpenAI request."""
    agent_context = _CONTEXT.get()
    if not agent_context:
        return kwargs

    request_kwargs = dict(kwargs)

    body = request_kwargs.get('extra_body')
    body = dict(body) if isinstance(body, dict) else {}
    nvext = body.get('nvext')
    nvext = dict(nvext) if isinstance(nvext, dict) else {}
    nvext['agent_context'] = dict(agent_context)
    body['nvext'] = nvext
    request_kwargs['extra_body'] = body

    headers = dict(request_kwargs.get('extra_headers') or {})
    headers.setdefault('x-request-id', str(uuid.uuid4()))
    request_kwargs['extra_headers'] = headers
    return request_kwargs


def _publish_record(record: Dict[str, Any]) -> None:
    publisher = _TOOL_EVENT_PUBLISHER
    if publisher is None:
        init_tool_event_publisher_from_env()
        publisher = _TOOL_EVENT_PUBLISHER
        if publisher is None:
            return

    try:
        publish = getattr(publisher, 'publish', None)
        if callable(publish):
            publish(record)
        else:
            publisher(record)
    except Exception:  # noqa
        pass


def _emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    record = {
        'schema': 'dynamo.agent.trace.v1',
        'event_type': event_type,
        'event_time_unix_ms': int(time.time() * 1000),
        'event_source': 'harness',
        **payload,
    }
    _publish_record(record)


def _tool_class(tool_name: str) -> str:
    if not tool_name:
        return 'unknown'
    if '---' in tool_name:
        return tool_name.split('---', 1)[0]
    if '/' in tool_name:
        return tool_name.split('/', 1)[0]
    return tool_name


def _tool_status(status: str) -> str:
    if status in {'ok', 'success'}:
        return 'succeeded'
    if status == 'timeout':
        return 'cancelled'
    return status


class ToolCallTrace:

    def __init__(self, tool_name: str, tool_call_id: Optional[str] = None):
        context = _CONTEXT.get()
        self.agent_context = dict(context) if context else None
        self.tool_name = tool_name or 'unknown'
        self.tool_call_id = tool_call_id or str(uuid.uuid4())
        self.tool_class = _tool_class(self.tool_name)
        self.started_at = time.perf_counter()
        self.start()

    def _tool_payload(self) -> Dict[str, Any]:
        return {
            'tool_call_id': self.tool_call_id,
            'tool_class': self.tool_class,
        }

    def start(self) -> None:
        if not self.agent_context:
            return
        tool = self._tool_payload()
        tool['status'] = 'running'
        _emit_event('tool_start', {
            'agent_context': self.agent_context,
            'tool': tool,
        })

    def end(
        self,
        status: str,
        error_type: Optional[str] = None,
    ) -> None:
        if not self.agent_context:
            return

        normalized_status = _tool_status(status)
        tool = self._tool_payload()
        tool.update({
            'duration_ms': int((time.perf_counter() - self.started_at) *
                               1000),
            'status': normalized_status,
        })
        if error_type:
            tool['error_type'] = error_type

        event_type = ('tool_end'
                      if normalized_status == 'succeeded' else 'tool_error')
        _emit_event(event_type, {
            'agent_context': self.agent_context,
            'tool': tool,
        })


def start_tool_call(tool_name: str,
                    tool_call_id: Optional[str] = None) -> ToolCallTrace:
    return ToolCallTrace(tool_name, tool_call_id)


def _reset_tool_event_publisher_for_tests() -> None:
    global _TOOL_EVENT_PUBLISHER, _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED
    with _TOOL_EVENT_PUBLISHER_LOCK:
        _TOOL_EVENT_PUBLISHER = None
        _TOOL_EVENT_PUBLISHER_INIT_ATTEMPTED = False
