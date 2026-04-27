# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


_CURRENT_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('ms_agent_current_trace_context', default=None))
_WORKFLOW_ID = os.environ.get('DYNAMO_AGENT_WORKFLOW_ID',
                              f'ms-agent-{uuid.uuid4().hex[:12]}')
_TRACE_PATH = os.environ.get('MS_AGENT_TRACE_JSONL')
_TRACE_LOCK = threading.Lock()


def build_agent_context(
    agent_tag: str,
    *,
    workflow_type_id: Optional[str] = None,
    parent_program_id: Optional[str] = None,
) -> Dict[str, Any]:
    workflow_type_id = (workflow_type_id
                        or os.environ.get('DYNAMO_AGENT_WORKFLOW_TYPE_ID')
                        or 'ms_agent')
    program_suffix = uuid.uuid4().hex[:8]
    context = {
        'workflow_type_id': workflow_type_id,
        'workflow_id': _WORKFLOW_ID,
        'program_id': f'{_WORKFLOW_ID}:{agent_tag}:{program_suffix}',
    }
    if parent_program_id:
        context['parent_program_id'] = parent_program_id
    return context


@contextlib.contextmanager
def activate_context(agent_context: Dict[str, Any]) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(dict(agent_context))
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_context() -> Optional[Dict[str, Any]]:
    ctx = _CURRENT_CONTEXT.get()
    return dict(ctx) if ctx else None


def current_program_id() -> Optional[str]:
    ctx = current_context()
    if not ctx:
        return None
    return ctx.get('program_id')


def merge_extra_body(extra_body: Any, agent_context: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(extra_body, dict):
        merged.update(extra_body)
    nvext = merged.get('nvext')
    if not isinstance(nvext, dict):
        nvext = {}
    else:
        nvext = dict(nvext)
    nvext['agent_context'] = dict(agent_context)
    merged['nvext'] = nvext
    return merged


class LLMRequestTrace:

    def __init__(self, agent_context: Dict[str, Any], *, model: str,
                 stream: bool, tool_count: int):
        self.agent_context = dict(agent_context)
        self.llm_call_id = str(uuid.uuid4())
        self.model = model
        self.stream = bool(stream)
        self.tool_count = tool_count
        self.started_at = time.perf_counter()
        self._linked = False
        self._ended = False
        self._dynamo_request_id: Optional[str] = None

    def _request_payload(self) -> Dict[str, Any]:
        payload = {
            'llm_call_id': self.llm_call_id,
            'model': self.model,
            'stream': self.stream,
            'tool_count': self.tool_count,
        }
        if self._dynamo_request_id:
            payload['dynamo_request_id'] = self._dynamo_request_id
        return payload

    def start(self) -> None:
        emit_event(
            'llm_request',
            {
                'agent_context': self.agent_context,
                'request': self._request_payload(),
            },
        )

    def link_response_id(self, response_id: Any) -> None:
        if self._linked or not response_id:
            return
        self._linked = True
        self._dynamo_request_id = str(response_id)
        emit_event(
            'llm_request_link',
            {
                'agent_context': self.agent_context,
                'request': self._request_payload(),
            },
        )

    def end(self, status: str, error: Optional[str] = None) -> None:
        if self._ended:
            return
        self._ended = True
        payload = self._request_payload()
        payload.update({
            'duration_ms': (time.perf_counter() - self.started_at) * 1000.0,
            'status': status,
        })
        if error:
            payload['error'] = error
        emit_event(
            'llm_request_end',
            {
                'agent_context': self.agent_context,
                'request': payload,
            },
        )


class _TracedStream:

    def __init__(self, stream: Iterable[Any], trace: LLMRequestTrace):
        self._stream = stream
        self._iterator = iter(stream)
        self._trace = trace

    def __iter__(self) -> '_TracedStream':
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._trace.end('ok')
            raise
        except Exception as exc:
            self._trace.end('error', str(exc))
            raise
        self._trace.link_response_id(getattr(chunk, 'id', None))
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def close(self) -> None:
        close = getattr(self._stream, 'close', None)
        if callable(close):
            close()
        self._trace.end('cancelled')


def start_llm_request(
        kwargs: Dict[str, Any], *, model: str, stream: bool,
        tool_count: int) -> Tuple[Dict[str, Any], Optional[LLMRequestTrace]]:
    agent_context = current_context()
    if not agent_context:
        return kwargs, None

    kwargs = dict(kwargs)
    kwargs['extra_body'] = merge_extra_body(
        kwargs.get('extra_body'), agent_context)
    trace = LLMRequestTrace(
        agent_context, model=model, stream=stream, tool_count=tool_count)
    trace.start()
    return kwargs, trace


def instrument_llm_request(kwargs: Dict[str, Any], *, model: str, stream: bool,
                           tool_count: int) -> Dict[str, Any]:
    kwargs, _ = start_llm_request(
        kwargs, model=model, stream=stream, tool_count=tool_count)
    return kwargs


def finish_llm_request(response: Any, trace: Optional[LLMRequestTrace],
                       *,
                       stream: bool) -> Any:
    if trace is None:
        return response
    if stream:
        return _TracedStream(response, trace)
    trace.link_response_id(getattr(response, 'id', None))
    trace.end('ok')
    return response


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


def _output_chars(output: Any) -> int:
    if isinstance(output, str):
        return len(output)
    return len(json.dumps(output, ensure_ascii=False, default=str))


class ToolCallTrace:

    def __init__(self, tool_name: str, tool_call_id: Optional[str] = None):
        self.agent_context = current_context()
        self.tool_name = tool_name or ''
        self.tool_call_id = tool_call_id or str(uuid.uuid4())
        self.started_at = time.perf_counter()

    def _tool_payload(self) -> Dict[str, Any]:
        return {
            'tool_call_id': self.tool_call_id,
            'tool_class': normalize_tool_class(self.tool_name),
            'tool_name_hash': hash_tool_name(self.tool_name),
        }

    def start(self) -> None:
        if not self.agent_context:
            return
        emit_event(
            'tool_start',
            {
                'agent_context': self.agent_context,
                'tool': self._tool_payload(),
            },
        )

    def end(self, status: str, output: Any = None) -> None:
        if not self.agent_context:
            return
        tool = self._tool_payload()
        tool.update({
            'duration_ms': (time.perf_counter() - self.started_at) * 1000.0,
            'status': status,
        })
        if output is not None:
            tool['output_chars'] = _output_chars(output)
        emit_event(
            'tool_end',
            {
                'agent_context': self.agent_context,
                'tool': tool,
            },
        )


def start_tool_call(tool_name: str,
                    tool_call_id: Optional[str] = None) -> ToolCallTrace:
    trace = ToolCallTrace(tool_name, tool_call_id)
    trace.start()
    return trace


def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    if not _TRACE_PATH:
        return
    event = {
        'schema': 'dynamo.agent.trace.v1',
        'event_type': event_type,
        'event_time_unix_ms': int(time.time() * 1000),
        'event_source': 'ms_agent',
        **payload,
    }
    line = json.dumps(event, ensure_ascii=False)
    with _TRACE_LOCK:
        with open(_TRACE_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
            f.write('\n')
