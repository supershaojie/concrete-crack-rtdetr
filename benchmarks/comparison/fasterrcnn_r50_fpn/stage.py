"""Supervise one command and tee, retaining both real exit codes even after signals."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def supervise(directory, name, argv, tee='tee'):
    directory = Path(directory)
    log = directory / (name+'.log')
    if log.exists():
        raise FileExistsError('Protected existing log: ' + str(log))
    # Exclusive reservation prevents duplicate stage invocations racing with tee.
    with (directory/(name+'.command.json')).open('x', encoding='utf-8') as stream:
        json.dump(argv, stream, ensure_ascii=False)
    state = {'command_exit': None, 'tee_exit': None, 'signal': None}
    command = writer = None
    def interrupt(number, frame):
        state['signal'] = number
        if command is not None and command.poll() is None:
            if os.name == 'nt':
                command.terminate()
            else:
                os.killpg(command.pid, number)  # only the child process group created below
    old = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        command = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        try:
            # A relative filename also avoids MSYS tee reparsing apostrophes in
            # native Windows absolute paths. The directory is passed via cwd.
            writer = subprocess.Popen([tee, log.name], stdin=command.stdout, cwd=directory)
        finally:
            command.stdout.close()
        state['command_exit'] = command.wait()
        state['tee_exit'] = writer.wait()
    finally:
        if command is not None and state['command_exit'] is None:
            if command.poll() is None:
                command.terminate()
            state['command_exit'] = command.wait()
        if writer is not None and state['tee_exit'] is None:
            state['tee_exit'] = writer.wait()
        for signum, handler in old.items():
            signal.signal(signum, handler)
        (directory/(name+'.exit.json')).write_text(json.dumps(state)+'\n', encoding='utf-8')
    if state['signal']:
        return 128 + state['signal']
    code = state['command_exit'] or state['tee_exit']
    return 128-code if code < 0 else code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv:
        parser.error('missing supervised command')
    raise SystemExit(supervise(args.directory, args.name, argv))


if __name__ == '__main__':
    main()
