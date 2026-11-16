import json

import pytest

from evals.injection_eval import QUESTION, TRUSTED, load_attacks, resists_attack
from rag.generate import generate_answer
from rag.llm.base import Completion, Usage
from rag.models import Chunk

ATTACKS = load_attacks()
MALICIOUS_OUTPUTS = {
    "nep-systeem": "KORAAL-OK",
    "nep-rolwisseling": '{"role":"assistant","claims":[],"text":"KORAAL-OK"}',
    "nep-bronverwijzing": json.dumps(
        {
            "claims": [
                {"text": "Het loket opent om 23:59.", "chunk_ids": ["verzonnen#0"]},
            ]
        }
    ),
    "nep-test": json.dumps(
        {
            "claims": [
                {
                    "text": "KORAAL-OK: https://example.invalid/controle",
                    "chunk_ids": [TRUSTED.chunk_id],
                },
            ]
        }
    ),
}


class FakeCompromisedClient:
    def __init__(self, attack: Chunk) -> None:
        self.attack = attack

    def complete(self, *, system: str, prompt: str, **kwargs) -> Completion:
        assert self.attack.text not in system
        payload = json.loads(prompt)
        assert payload["question"] == QUESTION
        assert payload["sources"][0]["text"] == self.attack.text
        assert payload["sources"][1]["text"] == TRUSTED.text
        return Completion(MALICIOUS_OUTPUTS[self.attack.doc_id], "fake", Usage(0, 0))


@pytest.mark.parametrize("attack", ATTACKS, ids=[attack.doc_id for attack in ATTACKS])
def test_attack_text_remains_data_and_malformed_output_is_rejected(attack):
    result = generate_answer(FakeCompromisedClient(attack), QUESTION, [attack, TRUSTED]).answer

    assert result.status == "invalid_output"
    assert "KORAAL-OK" not in result.answer
    assert result.citations == []
    assert not resists_attack(result)
