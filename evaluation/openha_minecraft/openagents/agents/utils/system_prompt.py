
with open("openagents/assets/system_prompt/grounding.txt", 'r') as f:
    GROUNDING_PROMPT = f.read()
    
with open("openagents/assets/system_prompt/motion.txt", 'r') as f:
    MOTION_PROMPT = f.read()

with open("openagents/assets/system_prompt/text_action.txt", 'r') as f:
    TEXT_ACTION_PROMPT = f.read()

with open("openagents/assets/system_prompt/grounding-coa.txt", 'r') as f:
    GROUNDING_COA_PROMPT = f.read()
    
with open("openagents/assets/system_prompt/motion-coa.txt", 'r') as f:
    MOTION_COA_PROMPT = f.read()

import json
import os
# gui_native system prompt has two sources, switched by env var:
#   GUI_USE_CLIENT_PROMPT=1  -> load the EXACT SYSTEM_PROMPT the SFT client uses
#                               (inference-server/prompts.py, incl. its key table),
#                               i.e. test with the model's training/deployment prompt verbatim.
#   otherwise (default)      -> the MC-adapted, data-driven template below
#                               (minecraft_keys.json -> {key_functions}).
_CLIENT_PROMPTS_PY = os.getenv(
    "GUI_CLIENT_PROMPTS_PY",
    "",
)
if os.getenv("GUI_USE_CLIENT_PROMPT", "0") == "1" and os.path.exists(_CLIENT_PROMPTS_PY):
    # exec prompts.py in its own dir (its data/*.json load is relative to __file__)
    _ns = {"__file__": _CLIENT_PROMPTS_PY}
    with open(_CLIENT_PROMPTS_PY, "r", encoding="utf-8") as f:
        exec(compile(f.read(), _CLIENT_PROMPTS_PY, "exec"), _ns)
    GUI_NATIVE_PROMPT = _ns.get("SYSTEM_PROMPT", "")
    if not GUI_NATIVE_PROMPT:
        raise ValueError(f"GUI_USE_CLIENT_PROMPT=1 but no SYSTEM_PROMPT found in {_CLIENT_PROMPTS_PY}")
else:
    # Minecraft key bindings, mirroring the client's prompts.py mechanism
    # (data/*.json -> {key_functions}). Single source of truth for the keys;
    # keep the `key` tokens in sync with GUI_TOKEN_TO_ENV in action_mapping.py.
    with open("openagents/assets/system_prompt/minecraft_keys.json", encoding="utf-8") as f:
        _mc_key_bindings = json.load(f)
    _mc_key_functions = "\n".join(f"- '{b['key']}'：{b['description']}" for b in _mc_key_bindings)
    with open("openagents/assets/system_prompt/gui_native.txt", 'r') as f:
        GUI_NATIVE_PROMPT = f.read().format(key_functions=_mc_key_functions)

# CPT (no-think) native-action prompt. Same action space / MC key bindings as
# gui_native, but the CPT model was trained WITHOUT <think> and without an
# instruction slot. Built unconditionally (independent of GUI_USE_CLIENT_PROMPT),
# reusing minecraft_keys.json. The task instruction, when enabled, is appended by
# vlm_client._generate_streaming (`你当前要完成的任务是：{text}`, "The task you need to complete now is: ..."), same as SFT.
with open("openagents/assets/system_prompt/minecraft_keys.json", encoding="utf-8") as f:
    _mc_key_bindings_cpt = json.load(f)
_mc_key_functions_cpt = "\n".join(f"- '{b['key']}'：{b['description']}" for b in _mc_key_bindings_cpt)
with open("openagents/assets/system_prompt/gui_native_cpt.txt", 'r') as f:
    GUI_NATIVE_CPT_PROMPT = f.read().format(key_functions=_mc_key_functions_cpt)

# English native-action prompt (gui_native_en): the English MC-adapted action-format
# prompt (includes its own "Common Keys in Minecraft" section) so PROMPT_LANG=en keeps
# the system prompt AND the task goal both in English. NOTE: the model was trained on
# the CHINESE prompt, so PROMPT_LANG=en is an OUT-OF-DISTRIBUTION probe, not the aligned
# config — expect lower SR than zh.
with open("openagents/assets/system_prompt/gui_native_en.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_EN_PROMPT = f.read()

# English SFT prompt: preserves gui_native's optional <think> contract.
with open("openagents/assets/system_prompt/gui_native_sft_en.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_SFT_EN_PROMPT = f.read().format(key_functions=_mc_key_functions)

# Prompt-following probe: identical to gui_native_sft_en except movement is bound to
# the arrow keys Up/Down/Left/Right and W/S/A/D are declared inert. Pair it with
# GuiActionTokenizer(key_map="arrows"|"both") so the decoder honours the rebinding.
with open("openagents/assets/system_prompt/gui_native_sft_en_arrows.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_SFT_EN_ARROWS_PROMPT = f.read()

# Same action space and optional-think contract as gui_native_sft_en, rewritten as an
# explicit grammar with worked positive/negative examples. Our checkpoints learned the
# format from training data and do not need it, but untrained commercial models read
# gui_native_sft_en's prose and reliably get three things wrong: they repeat the X Y Z
# triple inside every key group, emit anything from 5 to 11 groups instead of 6, and
# drop the <|action_start|> markers. Each wrong example here is one of those failures.
with open("openagents/assets/system_prompt/gui_native_strict_en.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_STRICT_EN_PROMPT = f.read()

# Legacy forced experiment protocol: every response must contain a non-empty
# reasoning block before the native GUI action. Special CPT now uses the same
# optional-thinking prompt as SFT.
with open("openagents/assets/system_prompt/gui_native_forced_thinking_en.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_FORCED_EN_PROMPT = f.read().format(key_functions=_mc_key_functions)
with open("openagents/assets/system_prompt/gui_native_forced_thinking_zh.txt", 'r', encoding="utf-8") as f:
    GUI_NATIVE_FORCED_ZH_PROMPT = f.read().format(key_functions=_mc_key_functions)

grounding_system_prompt = GROUNDING_PROMPT
motion_system_prompt = MOTION_PROMPT
rt2_system_prompt = "You are an AI agent performing tasks in Minecraft based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.\n## Action Space\n* camera move y,x # Move the camera or the cursor in GUI. \n* hotbar i #  Switch to hotbar item at index i.\n* forward <|reserved_special_token_191|> # Move forward.\n* back <|reserved_special_token_192|># Move backward.\n* right <|reserved_special_token_195|> # Strafe right.\n* left <|reserved_special_token_194|># Strafe left.\n* sprint <|reserved_special_token_197|> # Sprint in the current direction.\n* sneak <|reserved_special_token_198|> # Sneak; modifies movement in GUI and world.\n* use <|reserved_special_token_200|> # Use or place held item; in GUI, pick up/place one item.\n* drop <|reserved_special_token_202|> # Drop one item; Ctrl-drop to release full stack.\n* attack <|reserved_special_token_204|> #  Attack; in GUI, pick/place stacks or collect all similar items on double-click.\n* jump <|reserved_special_token_206|> # Jump\n* inventory <|reserved_special_token_177|> # Toggle inventory GUI.\nIf multiple actions are activated, connect without space.\n## Continuously take action until the task is completed. \n<|reserved_special_token_178|>raw actions<|reserved_special_token_179|>"
text_action_system_prompt = TEXT_ACTION_PROMPT
gui_native_system_prompt = GUI_NATIVE_PROMPT
gui_native_cpt_system_prompt = GUI_NATIVE_CPT_PROMPT
gui_native_en_system_prompt = GUI_NATIVE_EN_PROMPT
gui_native_sft_en_system_prompt = GUI_NATIVE_SFT_EN_PROMPT
# Reasoning protocol aliases used by the evaluator. These names deliberately
# describe the output contract rather than the checkpoint's CPT/SFT lineage.
gui_native_direct_zh_system_prompt = GUI_NATIVE_CPT_PROMPT
gui_native_direct_en_system_prompt = GUI_NATIVE_EN_PROMPT
gui_native_optional_zh_system_prompt = GUI_NATIVE_PROMPT
gui_native_optional_en_system_prompt = GUI_NATIVE_SFT_EN_PROMPT
gui_native_optional_en_arrows_system_prompt = GUI_NATIVE_SFT_EN_ARROWS_PROMPT
gui_native_optional_en_strict_system_prompt = GUI_NATIVE_STRICT_EN_PROMPT
gui_native_forced_zh_system_prompt = GUI_NATIVE_FORCED_ZH_PROMPT
gui_native_forced_en_system_prompt = GUI_NATIVE_FORCED_EN_PROMPT
mix_action_system_prompt = '''You are an AI agent performing tasks in Minecraft based on given instructions, history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.
## Action Space
You can choose different output action formats. For example, you may use a hierarchical format, which includes both a grounding-level action and a raw action; or a motion-level format along with a raw action. You can also choose to output only the raw action directly.
The action spaces for each level are as follows:
### grounding actions
* Kill <|object_ref_start|>mob<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Kill the Minecraft mobs.
* Mine <|object_ref_start|>block<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Destroy the Minecraft blocks.
* Approach <|object_ref_start|>object<|object_ref_end|><|point_start|>(x,y)<|point_end|> # approach to the target object.
* right click <|point_start|>(x,y)<|point_end|> # use the right lick button to interact with the object.
* move to <|point_start|>(x,y)<|point_end|> # Move the cursor in the GUI to the location of the object.
* no-op 
###motion action 
* move <direction> 
* sprint 
* sneak 
* turn <direction> 
* cursor move <direction> 
* jump 
* drop 
* swap 
* switch hotbar 
* open inventory 
* close inventory 
* close gui 
* attack / mine 
* place / use 
* operate gui 
* choose
### raw actions
* move('dx', 'dy') # Move the mouse position; dx and dy represent horizontal and vertical movement, respectively.
* click('left' or 'right')) # left click or right click the mouse
    - left_click # Attack; In GUI, pick up the stack of items or place the stack of items in a GUI cell; when used as a double click (attack - no attack - attack sequence), collect all items of the same kind present in inventory as a single stack.
    - right_click # Place the item currently held or use the block the player is looking at. In GUI, pick up the stack of items or place a single item from a stack held by mouse.
* press(keys) # press the keyboard buttons
    - 'w' # forward W key Move forward.
    - 's' # Move backward.
    - 'a' # Strafe left.
    - 'd' # Strafe right.
    - 'e' # Open or close inventory and the 2x2 crafting grid.
    - 'space' # Jump.
    - 'q' # Drop a single item from the stack of items the player is currently holding. If the player presses ctrl-Q then it drops the entire stack. In the GUI, the same thing happens except to the item the mouse is hovering over.\n       - '1'-'9' # Switch active item to the one in a given hotbar cell.
    - 'left.control' # Move fast in the current direction of motion.
    - 'left.shift' # Move carefully in current direction of motion. In the GUI it acts as a modifier key: when used with attack it moves item from/to the inventory to/from the hotbar, and when used with craft it crafts the maximum number of items possible instead of just 1.
* no_op # wait and do not interact with the world
If multiple actions are activated, use and connect.
## Continuously take action until the task is completed. 
Wrap your hierarchical action inside Skill: skill actions \nGrounding: grounding actions \nMotion: motion actions \nAction: raw actions
## User Instruction
'''

# Thinking-model variant of `text_action`. Identical action space and "Action: ..."
# output contract (so scores stay comparable to the OpenHA reference model), plus an
# explicit contract for models that reason before answering — Qwen3.5 thinks by default.
#
# Why this is needed: measured over 27066 steps of an unconstrained Qwen3.5 run,
# 33.8% of steps produced NO "Action:" line at all. The model rambled until it hit
# max_tokens, never closed </think>, so nothing parseable came out and the step was
# forced into a no_op — a third of the agent's steps wasted. It also averaged ~865
# decode tokens/step (vs ~20 for the SFT model), which is what made the eval take
# ~20h. Capping the reasoning and requiring the Action line fixes both at once
# (measured: ~31 tokens/step, with well-formed </think> + Action).
_TEXT_ACTION_THINK_RULES = '''## Thinking Format
You may reason before acting, but ALL reasoning must go inside <think> and </think>.
Keep the reasoning under 60 words: briefly note what you see and what to do next.
Immediately after </think>, output exactly ONE line with the action.
NEVER omit the Action line, and NEVER keep reasoning after </think>.

'''
# Insert before the trailing instruction header so the task goal still lands last.
text_action_think_system_prompt = text_action_system_prompt.replace(
    "## User Instruction", _TEXT_ACTION_THINK_RULES + "## User Instruction", 1)


def get_system_prompt(system_prompt_tag):
    if system_prompt_tag == "text_action":
        return text_action_system_prompt
    elif system_prompt_tag == "text_action_think":
        return text_action_think_system_prompt
    elif system_prompt_tag == "gui_native":
        return gui_native_system_prompt
    elif system_prompt_tag == "gui_native_cpt":
        return gui_native_cpt_system_prompt
    elif system_prompt_tag == "gui_native_en":
        return gui_native_en_system_prompt
    elif system_prompt_tag == "gui_native_sft_en":
        return gui_native_sft_en_system_prompt
    elif system_prompt_tag == "gui_native_direct_zh":
        return gui_native_direct_zh_system_prompt
    elif system_prompt_tag == "gui_native_direct_en":
        return gui_native_direct_en_system_prompt
    elif system_prompt_tag == "gui_native_optional_zh":
        return gui_native_optional_zh_system_prompt
    elif system_prompt_tag == "gui_native_optional_en":
        return gui_native_optional_en_system_prompt
    elif system_prompt_tag == "gui_native_optional_en_arrows":
        return gui_native_optional_en_arrows_system_prompt
    elif system_prompt_tag == "gui_native_optional_en_strict":
        return gui_native_optional_en_strict_system_prompt
    elif system_prompt_tag == "gui_native_forced_zh":
        return gui_native_forced_zh_system_prompt
    elif system_prompt_tag == "gui_native_forced_en":
        return gui_native_forced_en_system_prompt
    elif system_prompt_tag == "mix":
        return mix_action_system_prompt
    elif system_prompt_tag == "motion_action":
        return motion_system_prompt
    elif system_prompt_tag == "grounding_action":
        return grounding_system_prompt
    elif system_prompt_tag == "motion_coa":
        return MOTION_COA_PROMPT
    elif system_prompt_tag == "grounding_coa":
        return GROUNDING_COA_PROMPT
    else:
        raise ValueError
    
MODE_SYSTEM_PROMPT_MAP = {
    "greedy": {"mix", "motion_coa", "grounding_coa"},
    "text_action": {
        "text_action", "text_action_think",
        "gui_native", "gui_native_cpt", "gui_native_en", "gui_native_sft_en",
        "gui_native_direct_zh", "gui_native_direct_en",
        "gui_native_optional_zh", "gui_native_optional_en",
        "gui_native_optional_en_arrows", "gui_native_optional_en_strict",
        "gui_native_forced_zh", "gui_native_forced_en",
    },
    "grounding": {"grounding_action"},
    "motion": {"motion_action"},
}
