"""Tool execution helpers and TypedDict schemas describing external functions.

`doTool` runs a configured tool by spawning a subprocess and returns its exit code
and captured stdout/stderr. The `TOOL` table provides schema-like metadata used by callers.
"""

from asyncio import create_subprocess_exec, wait_for
from typing import TypedDict


class Property(TypedDict):
    type: str
    description: str


class Parameters(TypedDict):
    type: str
    properties: dict[str, Property]
    required: list[str]


class Function(TypedDict):
    name: str
    program: str
    description: str
    parameters: Parameters
    arguments: str


class Tool(TypedDict):
    type: str
    function: Function


async def doTool(tool: str, arg: str) -> dict[str, int | str]:
    """Execute a configured tool program with `arg` and capture output.

    Returns a dict with keys: `code` (exit code or 'timeout'), `stdout`, and `stderr`.
    """
    process = await create_subprocess_exec(*TOOL[tool]['function']['program'].split(), '-c', arg, stdout=-1, stderr=-1)
    assert process.stdout and process.stderr
    try:
        res = await wait_for(process.wait(), 60)
    except TimeoutError:
        process.kill()
        res = 'timeout'
    return {'code': res, 'stdout': (await process.stdout.read()).decode(errors='replace'), 'stderr': (await process.stderr.read()).decode(errors='replace')}

TOOL = {
    'python': Tool(type='function', function=Function(name='python', program='docker run --rm ipython/ipython ipython -c', description='Run code in python with 60s timeout and return stdout with stderr.', parameters=Parameters(type='object', properties={'arg': Property(type='string', description='The argument.')}, required=['arg']), arguments='')),
    'docker/python': Tool(type='function', function=Function(name='docker/python', program='ipython -c', description='Run code in python with 60s timeout and return stdout with stderr.', parameters=Parameters(type='object', properties={'arg': Property(type='string', description='The argument.')}, required=['arg']), arguments='')),
    'bash': Tool(type='function', function=Function(name='bash', program='docker run --rm ipython/ipython bash -c', description='Run code in python with 60s timeout and return stdout with stderr.', parameters=Parameters(type='object', properties={'arg': Property(type='string', description='The argument.')}, required=['arg']), arguments='')),
    'docker/bash': Tool(type='function', function=Function(name='docker/bash', program='bash -c', description='Run code in python with 60s timeout and return stdout with stderr.', parameters=Parameters(type='object', properties={'arg': Property(type='string', description='The argument.')}, required=['arg']), arguments='')),
}
