# DexDojo policy integration

This private copy starts at official XPolicyLab `72b3a52f2b069220088ef824ade5890885858379`, pinned by official RoboDojo `25691aa78fb34bbbe798aa3d880d57ef2788f696`. Original sources and licenses remain intact.

Keep model-specific conversion, training, deployment profiles, inference and scheduling inside `policy/<name>/`. `marvin.yml` supplies robot metadata and initial deployment settings for each of the 40 upstream adapters. These profiles require matching trained weights, normalization and native model assets. Native codecs and training configurations now preserve Marvin's complete arm/hand state. Robot-specific representations include 54D joints, 58D EE rotation6D, 52D EE deltas, and DM0/XR0's 66D combined pose/joint layouts. These are code adaptations, not 40-model checkpoint or physics acceptance.

- `process_data.sh` and `train.sh` remain the model-owned lifecycle entrypoints. Missing upstream stages are reported explicitly by the parent launcher, never replaced by no-op stubs.
- Wuji OpenPI training/configuration is in `policy/Pi_05/dexdojo`, entered from Pi_0 and Pi_05 train.sh; DP DDP training/data are in `policy/DP/dexdojo`, entered with `DP_TRAIN_MODE=ddp`.
- `setup_policy_server.py` loads `Model` or the model module's optional `create_model(config)` factory. No model-specific server branches are needed.
- Policy package initialization is lazy: importing deploy.py in the simulation environment does not import model frameworks.
- The WebSocket CALL extension forwards the allowlisted observation, action, batch and model lifecycle methods. `WsModelClient(direct_calls=True)` preserves per-step history and batch IDs; the previous INFER interface remains available to existing clients.
- Chunk length and scheduling belong to each policy; complete, finite robot actions are validated by the parent environment. Joint and end-effector representations are explicit. Scalar grippers are not expanded to Wuji hands.

See the parent DexDojo repository's `docs/MODELS_AND_TASK_SUITES.md` for commands, validation boundaries and fixed-suite rules. All-network training and real-physics policy acceptance require matching dependencies, checkpoints and datasets; deployment-loop tests use synthetic actions and are not learning evidence.

Xiaomi Robotics 1 upstream training is unavailable: its placeholders now exit nonzero. The parent inventory excludes it alongside the four adapters without train.sh. MolmoACT2 uses a version-checked, idempotent patch of its separately installed AllenAI LeRobot; Hy-VLA source provenance is recorded in its DEXDOJO_CHANGES.md. See the parent model guide for data requirements and validation limits.
