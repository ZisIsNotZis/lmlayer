from pydantic import BaseModel, Field
from time import time


class Session(BaseModel):
    user: str
    reasoning_content: str
    content: str
    cache_n: int
    prompt_n: int
    predicted_n: int
    prompt_ms: float
    predicted_ms: float
    tool: str
    arguments: str
    isTool: bool
    time: float = Field(default_factory=time)


_USER = dict[str, float]()
_SESSION = dict[str, dict[int, list[Session]]]()


async def getBalance(user: str) -> float:
    return _USER.setdefault(user, 1)


async def getSession(user: str, session: int) -> list[tuple[str, str, bool]]:
    return [(i.user, i.content, i.isTool)for i in _SESSION.get(user, {}).get(session, [])]


async def addWallet(user: str, value: float) -> None:
    _USER[user] = _USER.get(user, 0)+value


async def addSession(user: str, session: int, msg: Session) -> None:
    _SESSION.setdefault(user, {}).setdefault(session, []).append(msg)
