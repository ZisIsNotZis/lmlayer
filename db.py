from pydantic import BaseModel, Field
from time import time


class Session(BaseModel):
    role: str
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
    time: float = Field(default_factory=time)


_USER = dict[str, float]()
_SESSION = dict[str, dict[int, list[Session]]]()


async def hasBalance(user: str) -> bool:
    return _USER.setdefault(user, 1) > 0


async def getSession(user: str, session: int) -> list[tuple[str, str, str]]:
    return [(i.role, i.user, i.content)for i in _SESSION.get(user, {}).get(session, [])]


async def addWallet(user: str, value: float) -> None:
    _USER[user] = _USER.get(user, 0)+value


async def addSession(user: str, session: int, msg: Session) -> None:
    _SESSION.setdefault(user, {}).setdefault(session, []).append(msg)
