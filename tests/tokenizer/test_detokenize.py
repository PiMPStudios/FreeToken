from types import SimpleNamespace

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _Tok:
    eos_token_id = 0

    def batch_decode(self, sequences):
        return [self.decode(seq) for seq in sequences]

    def decode(self, ids):
        return ",".join(str(i) for i in ids)


def _msg(uid, tok, *, finished=False):
    return DetokenizeMsg(uid=uid, next_token=tok, finished=finished)


def test_multi_token_same_uid_is_incremental_not_overlapping():
    mgr = DetokenizeManager(_Tok(), eos_token_ids=frozenset({0}))
    chunks = mgr.detokenize([_msg(1, 10), _msg(1, 11), _msg(1, 12)])
    assert "".join(chunks) == "10,11,12"
    assert chunks[0] == "10"
    assert not chunks[1].startswith("10,11,12")


def test_distinct_uids_still_batch():
    mgr = DetokenizeManager(_Tok(), eos_token_ids=frozenset({0}))
    chunks = mgr.detokenize([_msg(1, 10), _msg(2, 20)])
    assert chunks == ["10", "20"]
