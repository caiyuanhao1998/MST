# Gradio demo

Run from the repository root with Python 3.11:

```bash
pip install -r requirements-demo.txt
python scripts/download_demo_assets.py
python app.py
```

Open <http://127.0.0.1:7876>. To choose a device or port, use `python app.py --device cpu --port 7877`.

Upload RGB or choose an example, select a model, and click **Reconstruct**. Click spectral thumbnails to switch the large preview. **Export selected channels** provides checkboxes for PNG export; the MAT download keeps all 28 raw bands. Click the large result image to inspect a pixel spectrum.

RGB inputs generate synthetic 28-band HSI and CASSI measurements before reconstruction with the official pretrained models. Native wavelengths follow `visualization/show_simulation.m`; editable labels and preview brightness do not change the raw reconstruction.
