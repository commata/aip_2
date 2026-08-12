# Training Record

- Created at: `2026-05-10T16:54:45`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `sac`

## Observation

- Mode: `tactical16`
- Size: `16`
- Description: Full tactical observation: ownship attitude + speed + altitude + health, relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. All features normalized to [-1, 1]. Observation space bounds: [-1, 1].
- Features:
  - `ownship_roll_norm`
  - `ownship_pitch_norm`
  - `ownship_yaw_norm`
  - `ownship_speed_norm`
  - `ownship_alt_norm`
  - `ownship_health_norm`
  - `delta_n_norm`
  - `delta_e_norm`
  - `delta_d_norm`
  - `ata_norm`
  - `aa_norm`
  - `az_norm`
  - `el_norm`
  - `target_health_norm`
  - `in_wez`
  - `pursuit_score_norm`

## Reward

- Description: Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) + damage differential + low altitude penalty + terminal rewards.
- Step penalty: `-0.01`
- Damage scale: `20.0`
- Pursuit scale: `0.3`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `0.1`
- Win reward: `100.0`
- Loss reward: `-100.0`
- Draw reward: `-30.0`

## CLI Arguments

```json
{
  "team_name": "team01",
  "version_tag": "student_sac_baseline",
  "algorithm": "sac",
  "iterations": 50,
  "print_every": 10,
  "use_tune": false,
  "checkpoint_frequency": 0,
  "notes": "Release student reward SAC baseline"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_DCS_base.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 300.0,
  "episode_step_limit": 18000,
  "reward": {
    "step_penalty": -0.01,
    "survival_bonus": 0.0,
    "pursuit_scale": 0.3,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "damage_scale": 20.0,
    "low_altitude_penalty": 0.1,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `3`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `4`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `5`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `6`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `7`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `8`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `9`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `10`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `11`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `12`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `13`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `14`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `15`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `16`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `17`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `18`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `19`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `20`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `21`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `22`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `23`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `24`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `25`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `26`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `27`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `28`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `29`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `30`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `31`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `32`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `33`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `34`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `35`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `36`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `37`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `38`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `39`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `40`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `41`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `42`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `43`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `44`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `45`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `46`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `47`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `48`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `49`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `50`: reward_mean=`n/a`, episode_len_mean=`n/a`
