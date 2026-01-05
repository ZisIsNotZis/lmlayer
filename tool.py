from pydantic import BaseModel
from asyncio import create_subprocess_exec, wait_for


class Property(BaseModel):
    type: str = 'string'
    description: str = 'The argument.'


class Parameters(BaseModel):
    type: str = 'object'
    properties: dict[str, Property] = {'arg': Property()}
    required: list[str] = ['arg']


class Function(BaseModel):
    name: str = ''
    description: str = ''
    parameters: Parameters = Parameters()
    arguments: str | None = None
    program: str | None = None

    async def run(self, arg) -> object:
        process = await create_subprocess_exec(self.program or self.name, '-c', arg, stdout=-1, stderr=-1)
        assert process.stdout and process.stderr
        try:
            arg = await wait_for(process.wait(), 60)
        except TimeoutError:
            process.kill()
            arg = 'timeout'
        return {'code': arg, 'stdout': (await process.stdout.read()).decode(errors='replace'), 'stderr': (await process.stderr.read()).decode(errors='replace')}


class Tool(BaseModel):
    type: str = 'function'
    function: Function


TOOL = {
    'python': Tool(function=Function(name='python', program='docker run --rm ipython/ipython ipython -c', description='Run code in python with 60s timeout and return stdout with stderr.')),
    'bash': Tool(function=Function(name='bash', program='docker run --rm ipython/ipython bash -c', description='Run code in bash with 60s timeout and return stdout with stderr.')),
    # 'search': Tool(function=Function(name='search', description='Search text in search engine.'))
}
