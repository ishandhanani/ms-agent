# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import queue
import unittest
from unittest import mock

from ms_agent import agent_trace


class FakePublisher:

    instances = []

    def __init__(self, endpoint: str, topic: str = ''):
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
        self.assertIn('started_at_unix_ms', records[0]['tool'])
        self.assertIn('started_at_unix_ms', records[1]['tool'])
        self.assertIn('ended_at_unix_ms', records[1]['tool'])
        self.assertIn('duration_ms', records[1]['tool'])
        self.assertGreaterEqual(records[1]['tool']['ended_at_unix_ms'],
                                records[1]['tool']['started_at_unix_ms'])
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

    def test_queue_tool_event_publisher_forwards_records(self):
        event_queue = queue.Queue()
        record = {
            'schema': 'dynamo.agent.trace.v1',
            'event_type': 'tool_end',
        }

        publisher = agent_trace.QueueToolEventPublisher(event_queue)
        publisher.publish(record)

        forwarded = event_queue.get_nowait()
        self.assertEqual(forwarded['type'], agent_trace.QUEUE_TOOL_EVENT_TYPE)
        self.assertIs(forwarded['record'], record)

    def test_publish_tool_event_record_uses_configured_publisher(self):
        publisher = FakePublisher('in-process')
        record = {
            'schema': 'dynamo.agent.trace.v1',
            'event_type': 'tool_end',
        }

        agent_trace.configure_tool_event_publisher(publisher)
        agent_trace.publish_tool_event_record(record)

        self.assertEqual(publisher.records, [record])


if __name__ == '__main__':
    unittest.main()
