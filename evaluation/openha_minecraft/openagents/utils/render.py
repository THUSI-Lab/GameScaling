import json
import cv2
from PIL import Image, ImageDraw, ImageFont
from openagents.assets import FONT_FILE
from openagents.utils import extract 
import numpy as np
from typing import (
    Optional, Sequence, List, Tuple, Dict, Union, Callable
)

IGNORE_PATTERNS = ['\n', '<|box_start|>', '<|box_end|>']
REPLACE_PATTERNS = {
    "\n": " ", 
    "<|box_start|>": "(", 
    "<|box_end|>": ")",
    "<|point_start|>": "(",
    "<|point_end|>": ")"
}

NOOP_ACTION = {"ESC": 0, "back": 0, "drop": 0, "forward": 0, "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0, "inventory": 0, "jump": 0, "left": 0, "right": 0, "sneak": 0, "sprint": 0, "swapHands": 0, "camera": [0, 0], "attack": 0, "use": 0, "pickItem": 0}

def auto_wrap_text(text, width, max_lines=6, ignore_patterns=IGNORE_PATTERNS, replace_patterns=REPLACE_PATTERNS):
    for ignore_pattern in ignore_patterns:
        text = text.replace(ignore_pattern, '')
    for replace_pattern, replace_text in replace_patterns.items():
        text = text.replace(replace_pattern, replace_text)
    wrapped_texts = []
    for i in range(0, len(text), width):
        wrapped_texts.append(text[i:i+width])
    return wrapped_texts[:max_lines]

def draw_texts(text, max_lines=8, max_width=38, width=640):
    image = Image.new('RGB', (width, 20 * max_lines), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Select font and size: system Arial font, size 16
    # font = ImageFont.truetype("./assets/fonts/SimHei.ttf", 16)
    if not FONT_FILE.exists():
        raise FileNotFoundError(
            f"Replay rendering needs the font {FONT_FILE}, which is not shipped. Download it from "
            "https://raw.githubusercontent.com/CraftJarvis/OpenHA/main/openagents/assets/fonts/SimHei.ttf "
            "or pass --no_render (see README.md).")
    font = ImageFont.truetype(FONT_FILE, 16)
    # max_width = 40  # Maximum displayable width; adjust to image width, leaving some margin

    if type(text) is str:
        lines = auto_wrap_text(text, max_width)[:max_lines]
    elif type(text) is list:
        lines = []
        for i in range(min(max_lines, len(text))):
            lines.append(text[i][:max_width])
    else:
        print("Upported text type except of list and str")
        return np.array(image)

    y = 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = 10
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += text_height + 5  # Add line spacing
    return np.array(image)

def draw_point(image, points):
    # Create a semi-transparent gray overlay
    overlay = image.copy()
    gray_overlay = np.full_like(image, 128, dtype=np.uint8)
    alpha = 0.5
    cv2.addWeighted(gray_overlay, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw the point on the image
    for point in points:
        x, y = point
        cv2.circle(overlay, (int(x), int(y)), radius=10, color=(255, 0, 0), thickness=-1)
        cv2.circle(overlay, (int(x), int(y)), radius=20, color=(0, 0, 255), thickness=1)

    # Save the image
    # image_path = f"output/temp/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{obj}-{str(uuid.uuid4())[:8]}.png"
    # cv2.imwrite(image_path, overlay)
    return overlay

def render_point_cot(frame,thought,radius = 5):
    if thought is None:
        return frame
    hierarchical_action = extract.extract_hierarchical_action(thought)

    # Then draw the point
    for pdx,p_item in enumerate(hierarchical_action["point"]):
        x_p, y_p = p_item["point"][0]
        caption_p = p_item["label"]

        # Draw a circle
        if pdx==0:
            circle_color =  (255, 0, 0)
        elif pdx==1:
            circle_color = (0, 200, 0)
        else:
            circle_color = (255,255,255)
        cv2.circle(frame, (x_p, y_p), radius, circle_color, -1)
    return frame

def render_grounding_point(frame, points, small_window_resolution:Tuple[np.int8]=None):
    if points is None:
        return frame 
    if not small_window_resolution:
        small_window_resolution = (240, 135)
    image_320x180 = None
    image_640x360 = frame
    resized_image = image_640x360.copy()
    if len(points) != 0:
        # Read the 320x180 image
        image_320x180 = image_640x360.copy()
        image_320x180 = draw_point(image_320x180, points)
        image_320x180 = cv2.resize(image_320x180, small_window_resolution)
    
    if image_320x180 is not None:
            # Overlay the 320x180 image onto the top-left corner of the resized image
            resized_image[10:small_window_resolution[1]+10, 10:small_window_resolution[0]+10] = image_320x180
            # Draw a rectangle at the top-left corner
            cv2.rectangle(resized_image, (10, 10), (small_window_resolution[0]+10, small_window_resolution[1]+10), (0, 255, 0), 2)
    return resized_image 

# Bounding boxes and points are rendered with a similar procedure
def render_grounding_box(frame, bbox):
    print("render bbox is not supported now")
    return frame 

# The task name is rendered at the bottom-left of the obs
def render_task_name(frame, task_name=None):
    if task_name == None:
        return frame
    text = task_name
    font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10
    text_y = 350 # 360 - 10 = image.shape[0] - 10
    # print(task_name)
    # print(frame.shape)
    # import ipdb; ipdb.set_trace()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# The thought is placed above the obs
def render_thought(frame, thought=None, max_lines=8, max_width=75):
    if thought is None:
        return frame
    render_text_image = draw_texts(thought, max_lines, max_width, width=frame.shape[1])
    render_image = np.concatenate([render_text_image, frame], axis=0)
    return render_image

def render_raw_action(frame, raw_action=None,max_lines=8, max_width=75):
    if raw_action is None:
        return frame
    # import ipdb; ipdb.set_trace()
    render_text_image = draw_texts(raw_action, max_lines=max_lines, max_width=75, width=frame.shape[1])
    render_image = np.concatenate([frame, render_text_image], axis=0)
    return render_image

"""
"attack": 0, "back": 0, "forward": 0, "jump": 0, "left": 0, "right": 0, "sneak": 0, "sprint": 0, "use": 0, "drop": 0, "inventory": 0, "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0, "camera": [10.0, -3.2153691330919005]
"""
KEEP_ENV_ACTION_KEYS = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 'drop', 'inventory']
def render_env_action(frame, env_action=None):
    if env_action is None:
        return frame
    # process the env_action
    new_env_action = {"hotbar": []}
    for i in range(1, 10):
        if env_action[f"hotbar.{i}"] == 1:
            new_env_action["hotbar"].append(f"hotbar.{i}")
    new_env_action["camera"] = [int(env_action["camera"][0]), int(env_action["camera"][1])]
    for key, value in env_action.items():
        if key in KEEP_ENV_ACTION_KEYS:
            new_env_action[key] = value
    
    font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    # font_color = (0, 0, 255)
    thickness = 1
    line_type = cv2.LINE_AA
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    text_x = 500
    text_y = 20
    for key, value in new_env_action.items():
        if key in ["hotbar", "camera"]:
            cv2.putText(frame, f"{value}", (text_x, text_y), font, font_scale, (255, 182, 193), thickness, line_type)
        else:
            if value == 1:
                cv2.putText(frame, f"{key}", (text_x, text_y), font, font_scale, (0, 0, 255), thickness, line_type)
            else:
                cv2.putText(frame, f"{key}", (text_x, text_y), font, font_scale, (0, 255, 0), thickness, line_type)
        text_y += 25
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# This function could be merged into render_raw_action
# def render_motion_action(frame, motion=None):
#     if motion == None:
#         return frame
#     text = motion
#     font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     font_color = (0, 0, 255)
#     thickness = 1
#     line_type = cv2.LINE_AA
#     text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
#     text_x = 10
#     text_y = 330
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
#     return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# This function could be merged into render_raw_action
# def render_language_action(frame, language_action=None):
#     if language_action is None:
#         return frame
#     # print(language_action)
#     if type(language_action) == str:
#         language_action = language_action[len('Action: '):].split(' and ')
#     # import ipdb; ipdb.set_trace()
#     render_text_image = draw_texts(language_action, max_lines=4, max_width=75)
#     render_image = np.concatenate([frame, render_text_image], axis=0)
#     return render_image

# FIXME: rewrite this function so that it can render saved json_action records
# The json_action is placed to the right of the obs
# def render_json_action(frame, json_action=None):
#     # print(f"render json_action is not supported now!")
#     if json_action is None:
#         return frame
#     render_text_image = draw_texts(json_action.replace('Action: ', '').split(' and '))
#     render_image = np.concatenate([frame, render_text_image], axis=0)
#     return render_image 

import os
import base64
import io
from openagents.utils.file_op import extract_frames_from_mp4

# def decode_base64(frame):
#     return np.frombuffer(base64.b64decode(frame['base64']), dtype=np.uint8)
def decode_obs(img_base64):
    '''
    Decode a Base64-encoded image string into a NumPy array.
    
    Args:
    img_base64: Base64-encoded image string
    
    Returns:
    numpy.array: decoded image array, in the same format as the original input
    '''
    # Decode the Base64 string into bytes
    img_bytes = base64.b64decode(img_base64)
    
    # Wrap the bytes in an in-memory file object
    buffer = io.BytesIO(img_bytes)
    
    # Open the image with PIL
    image = Image.open(buffer)
    
    # Convert the PIL image to a NumPy array
    obs = np.array(image)
    
    return obs

# the final merged version of renderring 
def render_video(
    instance_folder, 
    enable_task_name = True,
    enable_point=False,
    enable_rawaction = True,
    enable_thought = False,
    enable_envaction = False,
    enable_point_cot = False,
    small_window_resolution = None,
    thought_max_length = 8,
    raw_action_max_length = 8,
    ):
    # load the frames
    if os.path.exists(os.path.join(instance_folder, 'episode.jsonl')):
        frames = []
        with open(os.path.join(instance_folder, 'episode.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                frames.append(decode_obs(data['base64'])) # decode the base64 into numpy array
    elif os.path.exists(os.path.join(instance_folder, 'episode_1.mp4')):
        # load the mp4 
        # import ipdb; ipdb.set_trace()
        frames = extract_frames_from_mp4(os.path.join(instance_folder, 'episode_1.mp4'))
    else:
        raise ValueError(f"No episode.jsonl or episode_1.mp4 found in {instance_folder}")
    
    renderred_frames = frames.copy()
    # Render the task name if requested
    if enable_task_name:
        # load task name
        task_name = instance_folder.split('-')[-2].replace('kill_entity_', 'kill_entity:').replace('mine_block_', 'mine_block:')
        renderred_frames = [render_task_name(frame, task_name) for frame in renderred_frames]
            
    # Render the env action if requested
    if enable_envaction:
        if os.path.exists(os.path.join(instance_folder, 'action.jsonl')):
            env_actions = [NOOP_ACTION for _ in frames]
            with open(os.path.join(instance_folder, 'action.jsonl'), 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    # env_actions.append(data)
                    env_actions[i] = data
            renderred_frames = [render_env_action(frame, action) for frame, action in zip(renderred_frames, env_actions[1:])]
        else:
            print(f"No action.jsonl found in {instance_folder} for rendering env action")
            
    # Render the grounding point if requested
    if enable_point:
        # load the points
        if os.path.exists(os.path.join(instance_folder, 'raw_action.jsonl')):
            points = [[] for _ in frames]
            with open(os.path.join(instance_folder, 'raw_action.jsonl'), 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    # points.append(data.get('points', None))
                    points[i] = data.get('points', [])
            renderred_frames = [render_grounding_point(frame, point, small_window_resolution=small_window_resolution) for frame, point in zip(renderred_frames, points)]
        else:
            print(f"No raw_action.jsonl found in {instance_folder} for rendering grounding point")
            
    if enable_point_cot:
        if os.path.exists(os.path.join(instance_folder, 'raw_action.jsonl')):
            point_cots = []
            with open(os.path.join(instance_folder, 'raw_action.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    point_cot = data.get('thought', None)
                    if point_cot is None:
                        point_cot = data.get('raw_action', None)
                    point_cots.append(point_cot)
                renderred_frames = [render_point_cot(frame, thought = point_cot) for frame, point_cot in zip(renderred_frames, point_cots)]
    
    # Render the raw action if requested
    if enable_rawaction:
        # load the raw action
        if os.path.exists(os.path.join(instance_folder, 'raw_action.jsonl')):
            # raw_actions = []
            raw_actions = [" " for _ in frames]
            with open(os.path.join(instance_folder, 'raw_action.jsonl'), 'r') as f:
                # for line in f:
                for i, line in enumerate(f):
                    if i >= len(frames):
                        continue
                    data = json.loads(line)
                    # raw_actions.append(data.get('raw_action', None))
                    raw_actions[i] = data.get('raw_action', " ")
            renderred_frames = [render_raw_action(frame, raw_action) for frame, raw_action in zip(renderred_frames, raw_actions)]
        else:
            print(f"No raw_action.jsonl found in {instance_folder} for rendering raw action")
            
    # Render the thought if requested
    if enable_thought:
        if os.path.exists(os.path.join(instance_folder, 'thought.jsonl')):
            thoughts = ["" for _ in frames]
            with open(os.path.join(instance_folder, 'thought.jsonl'), 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    tick = data["tick"]
                    thoughts[tick] = data.get('thought', " ")
            renderred_frames = [render_thought(frame, thought,max_lines=thought_max_length) for frame, thought in zip(renderred_frames, thoughts)]
        elif os.path.exists(os.path.join(instance_folder, 'raw_action.jsonl')):
            thoughts = [" " for _ in frames]
            with open(os.path.join(instance_folder, 'raw_action.jsonl'), 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    # thoughts.append(data.get('thought', None))
                    thoughts[i] = data.get('thought', " ")
            renderred_frames = [render_thought(frame, thought,max_lines=thought_max_length) for frame, thought in zip(renderred_frames, thoughts)]
        else:
            print(f"No raw_action.jsonl found in {instance_folder} for rendering thought")
    
    return renderred_frames

