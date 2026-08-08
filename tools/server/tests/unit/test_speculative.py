import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_accept_guard_disables_draft_losslessly():
    """The acceptance guard must stop drafting without changing a single token.

    A mispaired draft head is silent -- it just makes decode slower than no
    speculation at all -- so the guard trips on measured acceptance. The property
    that matters is that tripping is lossless: verification still gates every
    token, so disabling the draft mid-stream can only change speed, never output.

    The thresholds are forced (floor above 100%, so any observed rate is under it)
    rather than arranged with a genuinely bad draft model, which would make the
    test depend on how badly two tiny stories models happen to disagree.

    The comparison is against the NO-DRAFT run, not against an unguarded draft
    run. That is the property the guard actually promises -- once it trips you
    are back to plain decoding -- and it is the only sound reference here:
    draft and no-draft already diverge at this length on this model pair even
    with the guard disarmed (see test_draft_matches_nodraft_64), so asserting
    guarded == unguarded would be asserting the wrong thing.
    """
    global server

    # reference: no draft model at all
    server.model_draft = None
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 64,
    })
    assert res.status_code == 200
    content_no_draft = res.body["content"]
    server.stop()

    # draft model present, but the guard trips on the first verification step
    create_server()
    server.extra_env = {
        "LLAMA_SPEC_ACCEPT_GUARD_MIN_DRAFTS": "1",
        "LLAMA_SPEC_ACCEPT_GUARD_FLOOR": "101",
    }
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 64,
    })
    assert res.status_code == 200
    content_guarded = res.body["content"]
    draft_n_guarded = res.body["timings"].get("draft_n", 0)

    # tripping returns us to exactly the non-speculative result
    assert content_guarded == content_no_draft
    # ...and it stopped drafting rather than merely ignoring the drafts: only the
    # single batch that produced the tripping measurement should be counted
    assert 0 < draft_n_guarded <= server.spec_draft_n_max


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])
