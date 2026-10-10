"""Offline tests: no model server, no game. `python -m pytest tests` or `python tests/test_core.py`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from gamescaling import (  # noqa: E402
    NOOP_ACTION_TEXT,
    PolicyConfig,
    build_request,
    build_system_prompt,
    finalize,
    parse_action,
    translate_controls,
)
from gamescaling.policy import encode_image  # noqa: E402


def test_parse_tagged_chunk():
    a = parse_action("<think>\n\n</think>\n\n<|action_start|>120 -40 0 ; W ; W LB ; ; Shift W ; ; one<|action_end|>")
    assert a.parsed and not a.truncated
    assert a.mouse == (120.0, -40.0, 0.0)
    assert a.steps == [["W"], ["W", "LB"], [], ["Shift", "W"], [], ["one"]]
    assert a.history_text == "<|action_start|>120 -40 0 ; W ; W LB ; ; Shift W ; ; one<|action_end|>"


def test_action_is_read_after_think_only():
    quoted = "<think>the format is <|action_start|>X Y Z ; k1 ; ; ; ; ; <|action_end|> so</think>"
    a = parse_action(quoted + "<|action_start|>5 0 0 ; D ; D ; ; ; ; <|action_end|>")
    assert a.parsed and a.mouse[0] == 5.0 and a.steps[0] == ["D"]


def test_unclosed_think_is_truncated_noop():
    a = parse_action("<think>I should move towards the <|action_start|>0 0 0 ; W")
    assert not a.parsed and a.truncated and a.is_noop
    assert a.history_text == NOOP_ACTION_TEXT


def test_clamp_vocab_and_key_limit():
    a = parse_action("<|action_start|>5000 -5000 9 ; A B C D E ; Foo W ; ; ; ; <|action_end|>")
    assert a.mouse == (1000.0, -1000.0, 5.0)
    assert a.steps[0] == ["A", "B", "C", "D"]
    assert a.steps[1] == ["W"] and a.dropped == ["Foo"]


def test_bare_body_and_placeholders():
    assert parse_action("10 0 0 ; W ; ; ; ; ;").parsed
    assert not parse_action("<|action_start|>X Y Z ; k1 ; ; ; ; ; <|action_end|>").parsed
    assert not parse_action("I see a monster on the left.").parsed


def test_lenient_recovers_skeleton():
    text = "<think>go right\n-5 4 ; D ; D ; D ; D ; D ; D"
    assert not parse_action(text).parsed
    a = parse_action(text, lenient=True)
    assert a.parsed and a.mouse[:2] == (-5.0, 4.0) and a.steps[0] == ["D"]


def test_finalize_restores_prefill_and_stop():
    out = finalize("\n\n</think>\n\n<|action_start|>0 0 0 ; W ; ; ; ; ; ", "stop", PolicyConfig())
    assert out.startswith("<think>") and out.endswith("<|action_end|>")
    assert parse_action(out).steps[0] == ["W"]
    out = finalize("\n\n<|action_start|>0 0 0 ; ; D ; ; ; ; ", "stop", PolicyConfig(reasoning=False))
    assert out.startswith("<think>\n\n</think>\n\n<|action_start|>")
    assert parse_action(out).steps[1] == ["D"]


def test_request_shape():
    cfg = PolicyConfig(model="m")
    assert cfg.image_size is None and cfg.reasoning
    body = build_request([{"role": "system", "content": "s"}], cfg)
    assert body["messages"][-1] == {"role": "assistant", "content": "<think>"}
    assert body["continue_final_message"] and not body["add_generation_prompt"]
    assert body["stop"] == ["<|action_end|>"] and body["skip_special_tokens"] is False
    body = build_request([{"role": "system", "content": "s"}], PolicyConfig(reasoning=False))
    assert body["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>"}
    assert body["continue_final_message"]


def test_image_encoding():
    img = Image.new("RGB", (1280, 720))
    assert encode_image(img, PolicyConfig()).startswith("data:image/jpeg;base64,")
    assert encode_image(img, PolicyConfig(image_format="png")).startswith("data:image/png;base64,")


def test_prompt_sections_in_order():
    p = build_system_prompt("Doom", rules="Kill monsters.", key_bindings={"W": "Forward", "LMB": "Fire"})
    order = ["currently playing Doom", "**Game Rules**", "**Common Keys in Doom**", "- LB -> Fire",
             "**Action only (usual)**", "**Explanation**", "**Key Naming Rules**"]
    idx = [p.index(s) for s in order]
    assert idx == sorted(idx)


def test_prompt_template_file_matches_builder():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "prompts", "system_prompt_template.txt")) as f:
        tpl = f.read().rstrip("\n")
    keys = {"W": "Move forward", "LB": "Fire"}
    bindings = "- W -> Move forward\n- LB -> Fire\n- Mouse X Y -> aim / rotate the camera (X>0 right, Y>0 down)"
    filled = (tpl.replace("{GAME_NAME}", "Doom").replace("{GAME_RULES}", "Kill monsters.")
              .replace("{KEY_BINDINGS}", bindings))
    assert filled == build_system_prompt("Doom", rules="Kill monsters.", key_bindings=keys)


def test_doom_example_matches_builder():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "prompts", "example_doom_battle1.txt")) as f:
        example = f.read().rstrip("\n")
    rules = example.split("**Game Rules**\n", 1)[1].split("\n**Common Keys in", 1)[0]
    keys = example.split("**Common Keys in Doom Battle-1**\n", 1)[1].split("\n**Output Format**", 1)[0]
    bindings = [line[2:].split(" -> ", 1) for line in keys.splitlines() if not line.startswith("- Mouse X Y")]
    assert example == build_system_prompt("Doom Battle-1", rules=rules, key_bindings=bindings)


def test_example_pair():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert Image.open(os.path.join(root, "examples", "doom_battle1.jpg")).size == (1280, 720)
    with open(os.path.join(root, "examples", "doom_battle1_action.txt")) as f:
        a = parse_action(f.read())
    assert a.parsed and a.mouse == (-37.0, -3.0, 0.0)
    assert all("LB" in g for g in a.steps)


def test_translate_controls():
    text = "- ArrowLeft: move left\n- a/d: strafe\n- 1-9: pick item\nPress Escape to pause."
    out = translate_controls(text)
    assert "- Left: move left" in out and "A/D" in out and "one-nine:" in out and "Esc to pause" in out


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
