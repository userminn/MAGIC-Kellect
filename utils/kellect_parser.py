"""Convert Kellect JSONL events into temporal graphs for MAGIC."""
import json
import os
import networkx as nx
import dgl


SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '..', '..', 'Kellect',
                                           'wechat_test', 'wechat_filtered.jsonl'))


def _node_key(kind, value):
    return '{}:{}'.format(kind, value)


def _event_target(event, args):
    name = event.get('Event', 'Unknown')
    lower = name.lower()
    if any(x in lower for x in ('file', 'image', 'load')):
        value = args.get('FileName') or args.get('OpenPath')
        if value:
            return 'file', value
    if any(x in lower for x in ('tcpip', 'udpip', 'dns')):
        address = args.get('daddr') or args.get('saddr')
        port = args.get('dport') or args.get('sport')
        if address:
            return 'network', '{}:{}'.format(address, port or '')
    if 'registry' in lower:
        value = args.get('KeyName') or args.get('KeyPath')
        if value:
            return 'registry', value
    if 'process' in lower or 'handle' in lower or 'alpc' in lower:
        value = args.get('TargetProcessId') or args.get('TargetProcessName')
        if value:
            return 'process', value
    tid = event.get('TID')
    if tid not in (None, -1, 0):
        return 'thread', '{}:{}'.format(event.get('PID', '?'), tid)
    # Preserve event type and activity volume when no concrete object is present.
    return 'activity', name


def _convert_window(events, node_types, edge_types):
    graph = nx.MultiDiGraph()
    node_ids = {}

    def add_node(kind, value):
        key = _node_key(kind, value)
        if key not in node_ids:
            node_ids[key] = len(node_ids)
            graph.add_node(node_ids[key], type=node_types[kind])
        return node_ids[key]

    for event in events:
        args = event.get('args') or {}
        pid = event.get('PID') or args.get('ProcessId') or args.get('SourceProcessId')
        process_name = event.get('PName') or args.get('TargetProcessName') or 'unknown'
        if pid is None:
            pid = 'unknown'
        process = add_node('process', '{}:{}'.format(pid, process_name))
        kind, value = _event_target(event, args)
        target = add_node(kind, value)
        event_type = event.get('Event', 'Unknown')
        graph.add_edge(process, target, type=edge_types[event_type])

    # Graphs are formed from events, so a valid window should always have edges.
    return dgl.from_networkx(graph, node_attrs=['type'], edge_attrs=['type'])


def load_kellect_graphs(window_size=256, max_windows=0):
    if window_size < 1:
        raise ValueError('window_size must be at least 1')
    if not os.path.isfile(SOURCE_PATH):
        raise FileNotFoundError('Kellect log not found: {}'.format(SOURCE_PATH))

    events = []
    with open(SOURCE_PATH, 'r', encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError('Invalid JSON on line {}: {}'.format(line_number, exc))

    if not events:
        raise ValueError('Kellect log contains no events')
    windows = [events[i:i + window_size] for i in range(0, len(events), window_size)]
    if max_windows > 0:
        windows = windows[:max_windows]
    if len(windows) < 3:
        raise ValueError('Need at least 3 graph windows; reduce --window_size or increase --max_windows')

    node_types = {name: i for i, name in enumerate(
        ['process', 'file', 'network', 'registry', 'thread', 'activity'])}
    event_names = sorted({event.get('Event', 'Unknown') for event in events})
    edge_types = {name: i for i, name in enumerate(event_names)}
    graphs = [_convert_window(window, node_types, edge_types) for window in windows]
    metadata = [{'window': i, 'first_timestamp': window[0].get('TimeStamp'),
                 'last_timestamp': window[-1].get('TimeStamp'), 'event_count': len(window)}
                for i, window in enumerate(windows)]
    return graphs, metadata
