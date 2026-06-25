# scenes

USD scenes built in the Isaac Sim GUI for THIS experiment live here, so they sync with the repo.

1. Open the editor: `./isaac-rubato` (from the repo root).
2. Build/edit the scene, then **File > Save As** into this folder, e.g. `scenes/my_scene.usd`.
3. Reference it from `../env.py` by path:
   ```python
   import os, isaaclab.sim as sim_utils
   SCENES = os.path.join(os.path.dirname(__file__), "scenes")
   spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(SCENES, "my_scene.usd"))
   ```
4. `git add` the `.usd` + `env.py`, commit. `git pull` on the lab machine and both are there.

`.usd` is binary; git handles it, but switch to git-lfs if scenes grow to tens of MB.
