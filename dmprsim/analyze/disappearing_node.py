"""
Draw a sequence diagram of all transmitted messages,
requires rx.msg.valid tracepoint
"""
import json
from pathlib import Path

from seqdiag import builder, drawer, parser as seq_parser

from dmprsim.analyze._utils.extract_messages import all_tracefiles, \
    extract_messages
from dmprsim.scenarios.disappearing_node import main as scenario

skel = """
   seqdiag {{
      activation = none;
      {}
   }}
"""


def main(args, results_dir: Path, scenario_dir: Path):
    scenario(args, results_dir, scenario_dir)

    if not getattr(args, 'sequence_diagram', False):
        return
    routers = set()
    messages = {}
    for router, tracefile in all_tracefiles([scenario_dir], 'rx.msg.valid'):
        routers.add(router)
        for time, message in extract_messages(tracefile):
            messages.setdefault(time, []).append((router, message))

    diag = []
    diag_skel = '{sender} -> {receiver} [label="{time}\n{type}\n{data}"]'
    for time in sorted(messages, key=float):
        for receiver, message in messages[time]:
            message = json.loads(message)
            sender = message['id']
            type = message['type']
            data = []
            if 'routing-data' in message:
                for policy in message['routing-data']:
                    for node, path in message['routing-data'][policy].items():
                        if path is not None:
                            path = path['path']
                        data.append('{}: {}'.format(node, path))
            data = '\n'.join(sorted(data))

            diag.append(diag_skel.format(sender=sender,
                                         receiver=receiver,
                                         type=type,
                                         data=data,
                                         time=time))

    diag.insert(0, ';'.join(sorted(routers)) + ';')
    result = skel.format('\n'.join(diag))

    tree = seq_parser.parse_string(result)
    diagram = builder.ScreenNodeBuilder.build(tree)
    filename = str(results_dir / 'sequence_diagram.svg')
    try:
        results_dir.mkdir(parents=True)
    except FileExistsError:
        pass
    draw = drawer.DiagramDraw(args.seq_diag_type, diagram, filename=filename)
    draw.draw()
    draw.save()
