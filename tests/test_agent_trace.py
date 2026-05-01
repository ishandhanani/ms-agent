# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import unittest
from unittest import mock

from ms_agent import agent_trace


class FakePublisher:

    instances = []

    def __init__(self, endpoint: str, topic: str = '', startup_delay_seconds=0):
        self.endpoint = endpoint
        self.topic = topic
        self.records = []
        FakePublisher.instances.append(self)

    def publish(self, record):
        self.records.append(record)


class TestDynamoAgentTrace(unittest.TestCase):

    def setUp(self):
        FakePublisher.instances.clear()
        agent_trace._reset_tool_event_publisher_for_tests()

    def tearDown(self):
        agent_trace._reset_tool_event_publisher_for_tests()

    def test_env_initializes_tool_event_publisher(self):
        env = {
            'DYNAMO_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT': 'tcp://127.0.0.1:20390',
            'DYNAMO_AGENT_TOOL_EVENTS_ZMQ_TOPIC': 'tools',
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                agent_trace, '_ZmqToolEventPublisher', FakePublisher):
            self.assertTrue(agent_trace.init_tool_event_publisher_from_env())

        self.assertEqual(len(FakePublisher.instances), 1)
        self.assertEqual(FakePublisher.instances[0].endpoint,
                         'tcp://127.0.0.1:20390')
        self.assertEqual(FakePublisher.instances[0].topic, 'tools')

    def test_tool_event_publish_lazily_initializes_from_env(self):
        env = {
            'DYNAMO_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT': 'tcp://127.0.0.1:20420',
        }
        context = {
            'workflow_id': 'run-1',
            'workflow_type_id': 'ms_agent',
            'program_id': 'run-1:agent',
        }

        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                agent_trace, '_ZmqToolEventPublisher', FakePublisher):
            with agent_trace.activate_context(context):
                trace = agent_trace.start_tool_call('web_search---query',
                                                    'call-1')
                trace.end('ok')

        self.assertEqual(len(FakePublisher.instances), 1)
        records = FakePublisher.instances[0].records
        self.assertEqual([record['event_type'] for record in records],
                         ['tool_start', 'tool_end'])
        self.assertEqual(records[0]['agent_context'], context)
        self.assertEqual(records[0]['tool']['tool_call_id'], 'call-1')
        self.assertEqual(records[0]['tool']['tool_class'], 'web_search')
        self.assertEqual(records[1]['tool']['status'], 'succeeded')

    def test_legacy_wrapper_env_name_is_supported(self):
        env = {
            'DYNAMO_AGENT_TRACE_TOOL_ZMQ_ENDPOINT': 'tcp://127.0.0.1:20391',
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                agent_trace, '_ZmqToolEventPublisher', FakePublisher):
            self.assertTrue(agent_trace.init_tool_event_publisher_from_env())

        self.assertEqual(FakePublisher.instances[0].endpoint,
                         'tcp://127.0.0.1:20391')


if __name__ == '__main__':
    unittest.main()
