# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal Dynamo agent trace integration helpers.

ms-agent only attaches Dynamo context to LLM requests and optionally publishes
tool lifecycle events. Dynamo owns normalized LLM tracing and trace sinks.
"""

import asyncio
import contextlib
import contextvars
import hashlib
import inspect
import json
import os
import time
import uuid
from typing import Any, Dict, Iterator, Optional

_CONTEXT: contextvars.ContextVar[Optional[Dict[str, str]]] = (
    contextvars.ContextVar('dynamo_agent_context', default=None))
_WORKFLOW_ID = os.environ.get('DYNAMO_AGENT_WORKFLOW_ID',
                              f'ms-agent-{uuid.uuid4().hex[:12]}')
_TOOL_EVENT_PUBLISHER: Optional[Any] = None


def configure_tool_event_publisher(publisher: Optional[Any]) -> None:
    """Register a best-effort publisher for Dynamo tool lifecycle events."""
    global _TOOL_EVENT_PUBLISHER
    _TOOL_EVENT_PUBLISHER = publisher


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


def current_context() -> Optional[Dict[str, str]]:
    context = _CONTEXT.get()
    return dict(context) if context else None


def current_program_id() -> Optional[str]:
    context = current_context()
    if not context:
        return None
    return context.get('program_id')


def merge_extra_body(
    extra_body: Any,
    agent_context: Dict[str, str],
) -> Dict[str, Any]:
    body = dict(extra_body) if isinstance(extra_body, dict) else {}
    nvext = body.get('nvext')
    nvext = dict(nvext) if isinstance(nvext, dict) else {}
    nvext['agent_context'] = dict(agent_context)
    body['nvext'] = nvext
    return body


def instrument_llm_request(
    kwargs: Dict[str, Any],
    *,
    model: str,
    stream: bool,
    tool_count: int,
) -> Dict[str, Any]:
    """Attach Dynamo context and x-request-id to an OpenAI request."""
    del model, stream, tool_count

    agent_context = current_context()
    if not agent_context:
        return kwargs

    request_kwargs = dict(kwargs)
    request_kwargs['extra_body'] = merge_extra_body(
        request_kwargs.get('extra_body'), agent_context)

    headers = dict(request_kwargs.get('extra_headers') or {})
    headers.setdefault('x-request-id', str(uuid.uuid4()))
    request_kwargs['extra_headers'] = headers
    return request_kwargs


async def _await_publish(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception:  # noqa
        pass


def _publish_record(record: Dict[str, Any]) -> None:
    publisher = _TOOL_EVENT_PUBLISHER
    if publisher is None:
        return

    try:
        publish = getattr(publisher, 'publish', None)
        result = publish(record) if publish is not None else publisher(record)
    except Exception:  # noqa
        return

    if not inspect.isawaitable(result):
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_publish(result))
    else:
        loop.create_task(_await_publish(result))


def _emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    record = {
        'schema': 'dynamo.agent.trace.v1',
        'event_type': event_type,
        'event_time_unix_ms': int(time.time() * 1000),
        'event_source': 'harness',
        **payload,
    }
    _publish_record(record)


def normalize_tool_class(tool_name: str) -> str:
    if not tool_name:
        return 'unknown'
    if '---' in tool_name:
        return tool_name.split('---', 1)[0]
    if '/' in tool_name:
        return tool_name.split('/', 1)[0]
    return tool_name


def hash_tool_name(tool_name: str) -> str:
    digest = hashlib.sha256(tool_name.encode('utf-8')).hexdigest()
    return f'sha256:{digest}'


def _output_bytes(output: Any) -> int:
    if isinstance(output, bytes):
        return len(output)
    if isinstance(output, str):
        return len(output.encode('utf-8'))
    try:
        encoded = json.dumps(output, ensure_ascii=False, default=str)
    except Exception:  # noqa
        encoded = str(output)
    return len(encoded.encode('utf-8'))


def _tool_status(status: str) -> str:
    if status in {'ok', 'success'}:
        return 'succeeded'
    if status == 'timeout':
        return 'cancelled'
    return status


class ToolCallTrace:

    def __init__(self, tool_name: str, tool_call_id: Optional[str] = None):
        self.agent_context = current_context()
        self.tool_name = tool_name or 'unknown'
        self.tool_call_id = tool_call_id or str(uuid.uuid4())
        self.tool_class = normalize_tool_class(self.tool_name)
        self.started_at = time.perf_counter()
        self.start()

    def _tool_payload(self) -> Dict[str, Any]:
        return {
            'tool_call_id': self.tool_call_id,
            'tool_class': self.tool_class,
            'tool_name_hash': hash_tool_name(self.tool_name),
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
        output: Optional[Any] = None,
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
        if output is not None:
            tool['output_bytes'] = _output_bytes(output)
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
