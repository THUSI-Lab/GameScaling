# Third-party notices

This evaluation package includes adapted third-party benchmark code and small scenario assets. Their upstream notices remain applicable; the repository's Apache-2.0 license does not replace them. Paths below are relative to the evaluation package.

| Component | Included material | Upstream notice |
|---|---|---|
| OpenHA | `openha_minecraft/` | MIT; Copyright (c) 2025 CraftJarvis. Original text: [OpenHA-MIT.txt](upstream-licenses/OpenHA-MIT.txt). [Source](https://github.com/CraftJarvis/OpenHA). |
| OmniGameArena | `bench_eval/omnigamearena/` | Apache License 2.0. Original text: [OmniGameArena-Apache-2.0.txt](upstream-licenses/OmniGameArena-Apache-2.0.txt). [Source](https://github.com/mxlin043/OmniGameArena). |
| GameWorld | `bench_eval/gameworld/` | Derived from [gameworld-project/GameWorld](https://github.com/gameworld-project/GameWorld). No standalone license file or license declaration was found in its README or project metadata at upstream commit `84f624bd1e120d30aeebf2eb42a39410e24d3bd4`. No additional license is assigned here. |
| Direct Future Prediction maps | `bench_eval/vizdoom/vizdoom_arena/scenarios/D3_battle.wad` and `D4_battle2.wad` | The files match [isl-org/DirectFuturePrediction/maps](https://github.com/isl-org/DirectFuturePrediction/tree/ea2e9c699b632dcbc423d680abaaf49533c9dfe1/maps). That repository's MIT license credits Intel Corporation (2017), Alexey Dosovitskiy and Vladlen Koltun; its original text is retained in [DirectFuturePrediction-MIT.txt](upstream-licenses/DirectFuturePrediction-MIT.txt). |
| ViZDoom scenarios | `bench_eval/vizdoom/vizdoom_arena/scenarios/defend_the_center.wad` and `defend_the_line.wad` | The files match [ViZDoom 1.2.4/scenarios](https://github.com/Farama-Foundation/ViZDoom/tree/1.2.4/scenarios). The upstream README states that original ViZDoom code is MIT-licensed and ZDoom contains code under different licenses. No scenario-specific license is asserted here. The upstream statement is retained in [ViZDoom-UPSTREAM-LICENSE-NOTE.txt](upstream-licenses/ViZDoom-UPSTREAM-LICENSE-NOTE.txt). |

The benchmark copies include evaluation-specific model clients, action adapters, protocol configurations, orchestration and result-processing changes. They are not unmodified upstream distributions. The archive does not identify the exact source commits from which these copies were derived; the links and hashes record the sources checked for these notices.

The UE5 game build, GameWorld browser-game assets, ViZDoom engine and Freedoom base IWAD, model weights, and SimHei font are not included in this evaluation code package. They are obtained separately and retain their respective upstream terms.

Source license texts and provenance were checked on 2026-11-11. [SOURCES.json](upstream-licenses/SOURCES.json) records the retrieved license-file URLs and Git blob hashes.
