# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlparse


_DASHSCOPE_HOST_MARKERS = ('dashscope', 'aliyuncs.com')


def _is_dashscope_endpoint(base_url: str) -> bool:
    host = urlparse(base_url or '').hostname or ''
    host = host.lower()
    return any(marker in host for marker in _DASHSCOPE_HOST_MARKERS)


def normalize_request_kwargs(base_url: str,
                             kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize provider-specific OpenAI-compatible request extensions."""
    extra_body = kwargs.get('extra_body')
    if not isinstance(extra_body, dict) or 'enable_thinking' not in extra_body:
        return kwargs

    if _is_dashscope_endpoint(base_url):
        return kwargs

    kwargs = dict(kwargs)
    extra_body = dict(extra_body)
    enable_thinking = extra_body.pop('enable_thinking')

    chat_template_kwargs = extra_body.get('chat_template_kwargs')
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    else:
        chat_template_kwargs = dict(chat_template_kwargs)
    chat_template_kwargs.setdefault('enable_thinking', enable_thinking)
    extra_body['chat_template_kwargs'] = chat_template_kwargs

    kwargs['extra_body'] = extra_body
    return kwargs
