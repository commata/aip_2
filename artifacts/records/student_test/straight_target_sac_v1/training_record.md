# Training Record

- Created at: `2026-07-14T14:53:10`
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
  "algorithm": "sac",
  "iterations": 200,
  "framework": "torch",
  "num_env_runners": 1,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "gamma": 0.99,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 10000,
  "model_fcnet_hiddens": "256,256",
  "model_fcnet_activation": "relu",
  "model_head_fcnet_hiddens": "",
  "model_head_fcnet_activation": "relu",
  "model_vf_share_layers": null,
  "network_spec_json": "",
  "use_lstm": false,
  "use_lstm_sac": false,
  "lstm_scope": "actor_only",
  "lstm_cell_size": 64,
  "max_seq_len": 8,
  "debug_io": false,
  "use_lstm_prioritized_replay": false,
  "output_name": "student_test",
  "output_tag": "straight_target_sac_v1",
  "notes": "Straight constant-speed target tracking & shootdown experiment (autopilot mode).",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 5,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 10,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 5,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\idos0\\Desktop\\AIpilot_gemini\\DogFightEnv\\Release\\experiments\\test_straight_target.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "ownship": [
    4000.0,
    0.0,
    -7000.0,
    0.0,
    0.0,
    0.0,
    300.0
  ],
  "target": [
    6000.0,
    0.0,
    -7000.0,
    0.0,
    0.0,
    0.0,
    250.0
  ],
  "initial_scenario": {
    "mode": "default",
    "legacy_use_random_scenario": false
  },
  "target_autopilot": {
    "heading_cmd": 0.0,
    "altitude_cmd": -7000.0,
    "speed_cmd": 250.0
  },
  "wez": {
    "angle_deg": 4.0,
    "min_range_m": 152.4,
    "max_range_m": 1500.0
  },
  "reward": {
    "mode": "default",
    "step_penalty": -0.01,
    "damage_scale": 20.0,
    "pursuit_scale": 0.3,
    "low_altitude_penalty": 0.1,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0
  },
  "reward_module": "student.my_reward"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `3`: reward_mean=`-292.9066603702332`, episode_len_mean=`380.0`
- iter `4`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `5`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`-316.5343450824956`, episode_len_mean=`418.0`
- iter `8`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `9`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `10`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `11`: reward_mean=`-283.1067593795154`, episode_len_mean=`342.0`
- iter `12`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `13`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `14`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `15`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `16`: reward_mean=`-354.4930970971157`, episode_len_mean=`500.0`
- iter `17`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `18`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `19`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `20`: reward_mean=`-357.75741834512945`, episode_len_mean=`391.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `23`: reward_mean=`-305.1906731796877`, episode_len_mean=`381.0`
- iter `24`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `25`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `26`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `27`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `28`: reward_mean=`-339.43159433785036`, episode_len_mean=`431.0`
- iter `29`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `30`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `31`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `32`: reward_mean=`-347.4705683366294`, episode_len_mean=`399.0`
- iter `33`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `34`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `35`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `36`: reward_mean=`-430.4359666584804`, episode_len_mean=`438.0`
- iter `37`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `38`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `39`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `40`: reward_mean=`-354.35703036190074`, episode_len_mean=`391.0`
- iter `41`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `42`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `43`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `44`: reward_mean=`-315.60667991557057`, episode_len_mean=`390.0`
- iter `45`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `46`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `47`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `48`: reward_mean=`-331.3893388040258`, episode_len_mean=`393.0`
- iter `49`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `50`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `51`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `52`: reward_mean=`-338.49700264998705`, episode_len_mean=`428.0`
- iter `53`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `54`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `55`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `56`: reward_mean=`-313.14563639652306`, episode_len_mean=`422.0`
- iter `57`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `58`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `59`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `60`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `61`: reward_mean=`-625.552270498336`, episode_len_mean=`488.0`
- iter `62`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `63`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `64`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `65`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `66`: reward_mean=`-368.3481621776118`, episode_len_mean=`518.0`
- iter `67`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `68`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `69`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `70`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `71`: reward_mean=`-300.36309149431594`, episode_len_mean=`423.0`
- iter `72`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `73`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `74`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `75`: reward_mean=`-312.21472994522645`, episode_len_mean=`417.0`
- iter `76`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `77`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `78`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `79`: reward_mean=`-355.97235451967475`, episode_len_mean=`444.0`
- iter `80`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `81`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `82`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `83`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `84`: reward_mean=`-318.63162266378515`, episode_len_mean=`490.0`
- iter `85`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `86`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `87`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `88`: reward_mean=`-327.57535014247946`, episode_len_mean=`438.0`
- iter `89`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `90`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `91`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `92`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `93`: reward_mean=`-364.7047395558798`, episode_len_mean=`462.0`
- iter `94`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `95`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `96`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `97`: reward_mean=`-428.73258489144536`, episode_len_mean=`437.0`
- iter `98`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `99`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `100`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `101`: reward_mean=`-322.8785330222324`, episode_len_mean=`347.0`
- iter `102`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `103`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `104`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `105`: reward_mean=`-382.6086584648452`, episode_len_mean=`431.0`
- iter `106`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `107`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `108`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `109`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `110`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `111`: reward_mean=`-249.3224929367422`, episode_len_mean=`600.0`
- iter `112`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `113`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `114`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `115`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `116`: reward_mean=`-702.4934579834437`, episode_len_mean=`487.0`
- iter `117`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `118`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `119`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `120`: reward_mean=`-320.2293636464161`, episode_len_mean=`375.0`
- iter `121`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `122`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `123`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `124`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `125`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `126`: reward_mean=`-165.29503993795132`, episode_len_mean=`600.0`
- iter `127`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `128`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `129`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `130`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `131`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `132`: reward_mean=`-79.31702959418823`, episode_len_mean=`600.0`
- iter `133`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `134`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `135`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `136`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `137`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `138`: reward_mean=`-108.83896675623876`, episode_len_mean=`600.0`
- iter `139`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `140`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `141`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `142`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `143`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `144`: reward_mean=`-136.71029941858245`, episode_len_mean=`600.0`
- iter `145`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `146`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `147`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `148`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `149`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `150`: reward_mean=`-262.541701424369`, episode_len_mean=`600.0`
- iter `151`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `152`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `153`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `154`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `155`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `156`: reward_mean=`-21.880822016448935`, episode_len_mean=`600.0`
- iter `157`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `158`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `159`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `160`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `161`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `162`: reward_mean=`-38.334506512681315`, episode_len_mean=`600.0`
- iter `163`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `164`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `165`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `166`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `167`: reward_mean=`-425.266466614935`, episode_len_mean=`562.0`
- iter `168`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `169`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `170`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `171`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `172`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `173`: reward_mean=`-57.7167872421442`, episode_len_mean=`600.0`
- iter `174`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `175`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `176`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `177`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `178`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `179`: reward_mean=`-59.68034778107018`, episode_len_mean=`600.0`
- iter `180`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `181`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `182`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `183`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `184`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `185`: reward_mean=`-101.75987475440361`, episode_len_mean=`600.0`
- iter `186`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `187`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `188`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `189`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `190`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `191`: reward_mean=`-195.5978304328173`, episode_len_mean=`600.0`
- iter `192`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `193`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `194`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `195`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `196`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `197`: reward_mean=`-137.57496111514965`, episode_len_mean=`600.0`
- iter `198`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `199`: reward_mean=`nan`, episode_len_mean=`nan`
