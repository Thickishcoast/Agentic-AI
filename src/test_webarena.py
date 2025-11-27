# src/test_webarena.py

from browser_env import ScriptBrowserEnv, create_id_based_action
import json
from pathlib import Path
import random

from agent_llm_policy import llm_policy, extract_text_from_obs


def parse_action(action_str: str):
    """Convert an action string like 'click [10]' into a WebArena action."""
    return create_id_based_action(action_str)


def serialize_info(step_info: dict) -> dict:
    """Make step_info JSON-safe by stringifying non-URL fields."""
    safe = {}
    for k, v in step_info.items():
        if k == "url":
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


def run_episode(config_file: str, max_steps: int = 5):
    task = "Click the 'Learn more' link if present on this page."

    env = ScriptBrowserEnv(
        headless=False,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
    )

    obs, info = env.reset(options={"config_file": config_file})
    print("Initial info:", info)
    print("Initial obs keys:", obs.keys())

    trajectory = []

    for step in range(max_steps):

        obs_text = extract_text_from_obs(obs)

        action_str = llm_policy(task, obs_text)

        print(f"Step {step}, chosen action: {action_str}")

        # 3) Parse and step environment
        action = parse_action(action_str)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        trajectory.append(
            {
                "step": step,
                "obs_text": obs_text,
                "action_str": action_str,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": serialize_info(step_info),
            }
        )

        obs = next_obs
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()
    return trajectory


def main():
    config_file = r"external\webarena\config_files\examples\0.json"
    traj = run_episode(config_file=config_file, max_steps=5)

    out_dir = Path("logged_episodes")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "episode_0.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(traj)} steps to {out_path}")


if __name__ == "__main__":
    main()
