# Gradio demo

Install the shared environment with `pip install -r requirements.txt`, then complete the dataset and model preparation in the README. The demo reuses `simulation/test_code/model_zoo/` and `datasets/TSA_simu_data/mask.mat`; all 10 RGB examples are included.

Run from the repository root:

```bash
python app.py
```

Open <http://127.0.0.1:7876>. To choose a device or port, use `python app.py --device cpu --port 7877`.

Upload RGB or choose an example, select a model, and click **Reconstruct**. Click spectral thumbnails to switch the large preview. **Export selected channels** provides checkboxes for PNG export; the MAT download keeps all 28 raw bands. Click the large result image to inspect a pixel spectrum.

RGB inputs generate synthetic 28-band HSI and CASSI measurements before reconstruction with the official pretrained models. Native wavelengths follow `visualization/show_simulation.m`; editable labels and preview brightness do not change the raw reconstruction.

For demo-only setup without the full datasets, `python scripts/download_demo_assets.py` prepares missing weights and the sensing mask. It skips existing files and supports the older `pretrained_weights/` directory. No extra downloads are required after the README preparation.
