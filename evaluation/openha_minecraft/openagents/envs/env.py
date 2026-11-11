import time
import json
import os
from openagents.envs.callbacks import RecordCallback, SummonMobsCallback,InitInventoryCallback,CommandsCallback
from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import SpeedTestCallback, RewardsCallback
from minestudio.simulator.callbacks.callback import MinecraftCallback


class MobsCallback(MinecraftCallback):
    """Observability: each tick, query nearby entities (Malmo MobAction, already in
    the action space next to VoxelAction) within a cube of half-size `box` around the
    player, and append the raw list to mobs.jsonl in the rollout dir. Lets us measure,
    offline, the distance + bearing to the target mob (does the agent approach / aim?).
    Gated by env var GUI_RECORD_MOBS=1 so normal eval runs are unaffected."""

    def __init__(self, record_path: str, box: int = 32):
        super().__init__()
        self.box = int(box)
        self._f = open(os.path.join(record_path, "mobs.jsonl"), "a", encoding="utf-8")

    def before_step(self, sim, action):
        b = self.box
        action["mobs"] = [-b, b, -b, b, -b, b]  # [x0,x1,y0,y1,z0,z1] cube around player
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self._f.write(json.dumps({"mobs": info.get("mobs", [])}, ensure_ascii=False) + "\n")
        self._f.flush()
        return obs, reward, terminated, truncated, info


def env_init(
        task_config:dict, 
        rollout_path:str, 
        args:dict,
        obs_size:tuple= (224,224), 
        render_size:tuple= (640, 360),
        camera_cfg:CameraConfig=None,
        record_raw_action:bool=True
    ) -> MinecraftSim:
    '''
    description: Build and configure the environment
    return {*}
    '''
    
    # Native engine resolution. Evaluation wrappers leave this at 640x360 and
    # resize only at the VLM boundary; these overrides exist solely for explicit
    # engine-resolution ablations. obs_size is the auxiliary-policy input.
    rw = getattr(args, "render_width", None)
    rh = getattr(args, "render_height", None)
    if rw and rh:
        render_size = (int(rw), int(rh))

    seed = task_config['seed']
    init_actions = task_config['init_actions']
    # callbacks
    init_inventory = task_config['callback']['init_inventory'].get("init_inventory",[])
    commands = task_config['callback'].get('commands',[])
    reward_cfg = task_config['rewards']
    callbacks = [
        RecordCallback(record_path=rollout_path, fps=args.fps, frame_type="pov"),
        InitInventoryCallback(init_inventory,
            inventory_distraction_level=task_config['callback']['init_inventory'].get("inventory_distraction_level",[0]),
            equip_distraction_level=task_config['callback']['init_inventory'].get("equip_distraction_level",[0]),
            forbidden_slots = task_config['callback']['init_inventory'].get("forbidden_slots",[]),
        ),
        
        RewardsCallback(reward_cfg),
        CommandsCallback(commands),
    ]
    mobs = task_config['callback'].get("mobs")
    if mobs:
        callbacks.append(SummonMobsCallback(mobs))
    # Observability (opt-in): log nearby entities per tick for target distance/bearing.
    if os.getenv("GUI_RECORD_MOBS"):
        callbacks.append(MobsCallback(rollout_path, box=int(os.getenv("GUI_MOBS_BOX", "32"))))
    
    env = MinecraftSim(
        action_type="env", 
        obs_size = obs_size, 
        render_size = render_size,
        seed = seed,
        preferred_spawn_biome=None,
        camera_config = camera_cfg,
        callbacks=callbacks,
    )
    obs, info = env.reset()
    flag = False
    
    for action in init_actions:
        time.sleep(0.1)
        obs, reward, terminated, truncated, info = env.step(action)
    
    if record_raw_action:
        raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
        with open(raw_action_file_path, 'a', encoding='utf-8') as f:
            for _ in range(len(init_actions)):
                f.write(json.dumps({"raw_action":""}, ensure_ascii=False) + '\n')
        
    return env
