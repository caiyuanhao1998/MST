"""Local Gradio demo: RGB → synthetic HSI → CASSI measurement → MST model zoo."""

import argparse
import getpass
import logging
import os
import socket
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".local/gradio-cache"))

import gradio as gr
import numpy as np
import torch

from mst_demo.artifacts import cleanup_result, export_result, spectrum_figure
from mst_demo.engine import Reconstructor
from mst_demo.models import MODELS
from mst_demo.spectral import (
    DEFAULT_CHANNEL,
    DEFAULT_WAVELENGTH_TEXT,
    WAVELENGTHS,
    band_image,
    channel_gallery,
    parse_wavelengths,
)

LOG = logging.getLogger(__name__)
CSS = """
.gradio-container {max-width:1240px !important; width:100% !important;
 min-width:0 !important; margin:auto;}
footer {display:none !important;}
#hero h1 {font-size:34px; letter-spacing:-.7px; margin-bottom:4px;}
#reconstruct {min-height:44px;}
#input-column, #result-column {gap:12px;}
#channel-strip-label p, #example-strip-label p {margin:0; font-size:13px;}
#rgb-input .image-container, #reconstruction-preview .image-container,
#rgb-input .image-frame, #reconstruction-preview .image-frame {
 width:100% !important; height:100% !important;
}
#rgb-input .image-frame img, #reconstruction-preview .image-frame img {
 width:100% !important; height:100% !important; object-fit:contain !important;
}
:is(#spectral-gallery, #sci-examples) {min-height:116px; width:100% !important; min-width:0 !important; overflow:hidden;}
:is(#spectral-gallery, #sci-examples) .grid-wrap {width:100%; min-width:0; max-width:100%; overflow-x:auto !important; overflow-y:hidden !important;
 height:114px !important; padding:8px !important; box-sizing:border-box; scrollbar-width:thin;}
:is(#spectral-gallery, #sci-examples) .grid-container {display:grid !important; grid-auto-flow:column !important;
 grid-template-columns:none !important; grid-auto-columns:86px !important;
 grid-template-rows:92px !important; grid-auto-rows:92px !important;
 gap:8px !important; width:max-content !important; min-width:100%;}
:is(#spectral-gallery, #sci-examples) .gallery-item {width:86px !important; height:92px !important;}
:is(#spectral-gallery, #sci-examples) .caption-label {left:4px; right:4px; bottom:3px; padding:2px;
 max-width:calc(100% - 8px); font-size:10px; line-height:1.2; white-space:normal; text-align:center;}
:is(#spectral-gallery, #sci-examples) .thumbnail-lg:hover .caption-label {opacity:1;}
#export-help p {font-size:13px;}
:is(#spectral-gallery, #sci-examples) .gallery-item {border-radius:7px !important; box-shadow:none !important;}
#example-strip-label, #channel-strip-label {min-height:20px;}
#model-panel, #preview-panel {min-height:100px; padding:12px;
 background:var(--block-background-fill) !important; border:1px solid var(--border-color-primary) !important;
 border-radius:10px !important;}
#model-controls, #preview-controls {gap:12px; min-height:74px; align-items:center;}
#model-controls {align-items:flex-end;}
:is(#model-panel, #preview-panel) .form {border:0; box-shadow:none; background:transparent;}
:is(#model-panel, #preview-panel) .styler {background:transparent !important;}
:is(#model-panel, #preview-panel) [data-testid="block-info"] {
 background:transparent; color:var(--body-text-color); padding:0;
 font-size:13px; font-weight:600; line-height:18px;}
#model-dropdown .wrap:not(.default) {min-height:40px; border:1px solid var(--input-border-color) !important;
 border-radius:7px !important;}
:is(#model-dropdown, #display-brightness, #pseudocolor) {border:0; padding:0; background:transparent;}
#refresh-weights {white-space:nowrap; min-height:40px; height:40px; align-self:flex-end;
 border:1px solid var(--border-color-primary) !important; border-radius:8px !important;}
#action-row {gap:12px;}
#reconstruct, #clear-images {height:44px; min-height:44px;}
#run-status {min-height:44px; padding:10px 12px; border:1px solid var(--border-color-primary) !important;
 border-radius:8px; background:var(--block-background-fill) !important;}
#run-status p {font-size:12px; line-height:22px; margin:0;}
#export-panel, #wavelength-settings {min-height:44px; padding:10px 12px;}
:is(#export-panel, #wavelength-settings) > .label-wrap {min-height:22px; align-items:center;}
:is(#measurement-panel, #spectrum-panel) {padding:12px;}
:is(#measurement-panel, #spectrum-panel) > .label-wrap {min-height:24px; align-items:center;}
:is(#export-panel, #wavelength-settings, #measurement-panel, #spectrum-panel) > .label-wrap {
 gap:12px; font-size:13px; font-weight:600;}
#spectrum-plot {height:260px !important; min-height:260px;}
:is(#rgb-input, #reconstruction-preview, #measurement-image, #mask-image) > label {
 max-width:calc(100% - 68px); white-space:normal; line-height:1.3;}
:is(#rgb-input, #reconstruction-preview, #measurement-image, #mask-image) > label > span {flex-shrink:0;}
#spectrum-plot .matplotlib {height:100% !important;}
#spectrum-plot img {max-width:100%; max-height:100%; object-fit:contain;}
#device-info p {font-size:12px; color:var(--body-text-color-subdued); margin:0;}

@media (max-width:900px) {
 .row {flex-direction:column !important;}
 .row > * {min-width:0 !important; width:100% !important; flex-basis:auto !important;}
 img {max-width:100%;}
 #hero h1 {font-size:27px;}
 #model-panel, #preview-panel {min-height:0;}
 #model-controls, #preview-controls {min-height:0;}
 #spectrum-plot {height:auto !important; min-height:160px;}
 #spectrum-plot .matplotlib {height:auto !important;}
 #run-status p {line-height:20px;}

}
"""


def build_demo(engine):
    native_channels = list(range(1, 29))
    default_wavelength = float(WAVELENGTHS[DEFAULT_CHANNEL - 1])
    choices = engine.available_models()
    initial = (
        "mst_s"
        if any(key == "mst_s" for _, key in choices)
        else (choices[0][1] if choices else None)
    )
    output_root = ROOT / ".local/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    with gr.Blocks(
        title="MST · CVPR 2022",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate",
            font=["Arial", "sans-serif"],
            font_mono=["Consolas", "monospace"],
        ).set(
            block_border_width="1px",
            block_border_width_dark="1px",
            block_radius="10px",
            block_shadow="none",
            block_shadow_dark="none",
            block_label_background_fill="*neutral_50",
            block_label_background_fill_dark="*neutral_900",
            block_label_text_color="*neutral_700",
            block_label_text_color_dark="*neutral_200",
            block_label_text_size="13px",
            block_label_text_weight="600",
            block_label_shadow="none",
            input_radius="7px",
            button_border_width="1px",
            button_border_width_dark="1px",
            button_large_text_size="14px",
            button_large_text_weight="600",
            button_large_padding="10px 16px",
            button_large_radius="8px",
            button_small_radius="8px",
            button_secondary_shadow="none",
            button_secondary_shadow_hover="none",
            button_secondary_shadow_active="none",
            button_primary_shadow="none",
            button_primary_shadow_hover="none",
            button_primary_shadow_active="none",
        ),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# MST · CVPR 2022", elem_id="hero")
        gr.Markdown(
            "Upload an RGB image, choose a model, and click **Reconstruct**. "
            "Click a spectral thumbnail to inspect a band."
        )
        state = gr.State(value=None, time_to_live=3600, delete_callback=cleanup_result)
        band = gr.State(value=DEFAULT_CHANNEL)
        with gr.Row(equal_height=False, elem_id="comparison-row"):
            with gr.Column(scale=1, min_width=320, elem_id="input-column"):
                source = gr.Image(
                    label="RGB input",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                    height=370,
                    elem_id="rgb-input",
                    show_share_button=False,
                )
                examples = sorted((ROOT / "demo/examples/simulation").glob("scene_*.png"))
                example_gallery = None
                if examples:
                    gr.Markdown("**10 Official SCI RGB examples**", elem_id="example-strip-label")
                    example_gallery = gr.Gallery(
                        value=[(str(path), f"Scene {i:02d}") for i, path in enumerate(examples, 1)],
                        label="10 Official SCI RGB examples",
                        show_label=False,
                        columns=len(examples),
                        rows=1,
                        height=116,
                        object_fit="contain",
                        format="png",
                        interactive=False,
                        allow_preview=False,
                        preview=False,
                        show_share_button=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        elem_id="sci-examples",
                    )
                with gr.Group(elem_id="model-panel"):
                    with gr.Row(elem_id="model-controls"):
                        model = gr.Dropdown(
                            choices=choices,
                            value=initial,
                            scale=4,
                            label="Reconstruction model",
                            container=True,
                            elem_id="model-dropdown",
                        )
                        refresh = gr.Button(
                            "Refresh Weights", scale=1, min_width=160, elem_id="refresh-weights"
                        )
                with gr.Row(elem_id="action-row"):
                    run = gr.Button(
                        "Reconstruct", variant="primary", scale=3, elem_id="reconstruct"
                    )
                    clear = gr.Button("Clear", scale=1, elem_id="clear-images")
                status = gr.Markdown("Ready. Upload an RGB image to begin.", elem_id="run-status")
            with gr.Column(scale=1, min_width=320, elem_id="result-column"):
                overview = gr.Image(
                    label=f"Reconstructed HSI · Ch {DEFAULT_CHANNEL} · {default_wavelength:g} nm",
                    elem_id="reconstruction-preview",
                    interactive=False,
                    height=370,
                    format="png",
                    show_share_button=False,
                )
                gr.Markdown(
                    "**Reconstructed spectral channels** · scroll and click to preview",
                    elem_id="channel-strip-label",
                )
                gallery = gr.Gallery(
                    label="Reconstructed spectral channels",
                    show_label=False,
                    columns=28,
                    rows=1,
                    height=116,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                    allow_preview=False,
                    preview=False,
                    show_share_button=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    elem_id="spectral-gallery",
                )
                with gr.Group(elem_id="preview-panel"):
                    with gr.Row(elem_id="preview-controls"):
                        brightness = gr.Slider(
                            0.1,
                            5,
                            value=1,
                            step=0.05,
                            scale=3,
                            label="Preview brightness",
                            container=True,
                            elem_id="display-brightness",
                        )
                        color = gr.Checkbox(
                            value=True,
                            label="Pseudocolor",
                            scale=1,
                            container=False,
                            elem_id="pseudocolor",
                        )
                with gr.Accordion("Export selected channels", open=False, elem_id="export-panel"):
                    export_channels = gr.CheckboxGroup(
                        choices=[(f"Ch {i:02d}", i) for i in native_channels],
                        value=native_channels,
                        label="Channels to export",
                        info="Tick channel numbers below. Thumbnail clicks only change the preview.",
                        elem_id="export-channel-selection",
                    )
                    with gr.Row():
                        select_all = gr.Button("Select all", size="sm")
                        select_none = gr.Button("Select none", size="sm")
                        select_current = gr.Button("Current channel only", size="sm")
                    export_help = gr.Markdown(
                        "Reconstruct an image first to enable export.", elem_id="export-help"
                    )
                    export_button = gr.Button(
                        "Export selected channels",
                        variant="primary",
                        interactive=False,
                        elem_id="export-channels-button",
                    )
                    with gr.Row():
                        zip_file = gr.File(
                            label="Selected PNG channels (ZIP)",
                            interactive=False,
                            height=85,
                            elem_id="channel-zip",
                        )
                        mat_file = gr.File(
                            label="Full HSI cube (MAT)",
                            interactive=False,
                            height=85,
                            elem_id="full-cube-mat",
                        )
                    gr.Markdown(
                        "Choose checkboxes → **Export selected channels** → download the ZIP. "
                        "The MAT file always contains all 28 raw bands."
                    )
                with gr.Accordion(
                    "Wavelength labels (advanced)", open=False, elem_id="wavelength-settings"
                ):
                    wavelength_text = gr.Textbox(
                        value=DEFAULT_WAVELENGTH_TEXT,
                        lines=3,
                        label="Wavelength labels (nm)",
                        elem_id="wavelength-labels",
                    )
                    gr.Markdown(
                        "Labels and pseudocolors are editable. Models keep their 28 trained bands."
                    )
                    with gr.Row():
                        apply_settings = gr.Button("Apply wavelength labels")
                        reset_settings = gr.Button("Reset repository defaults")
        with gr.Row(equal_height=False, elem_id="diagnostics-row"):
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion(
                    "Measurement and sensing mask", open=True, elem_id="measurement-panel"
                ):
                    with gr.Row():
                        measurement_image = gr.Image(
                            label="CASSI measurement",
                            height=260,
                            interactive=False,
                            format="png",
                            show_share_button=False,
                            elem_id="measurement-image",
                        )
                        mask_image = gr.Image(
                            label="Sensing mask",
                            height=260,
                            interactive=False,
                            format="png",
                            show_share_button=False,
                            elem_id="mask-image",
                        )
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion(
                    "Spectrum · click the large result image to inspect a pixel",
                    open=True,
                    elem_id="spectrum-panel",
                ):
                    spectrum = gr.Plot(
                        label="Synthetic and reconstructed spectra",
                        show_label=False,
                        elem_id="spectrum-plot",
                    )
        gr.Markdown(
            f"**Device:** `{engine.device}` · All inference and downloads remain local.",
            elem_id="device-info",
        )

        outputs = [
            overview,
            gallery,
            measurement_image,
            mask_image,
            spectrum,
            mat_file,
            zip_file,
            export_help,
            export_button,
            status,
            state,
        ]
        view_outputs = [overview, gallery, spectrum, mat_file, zip_file, export_help, export_button]
        serial = {"concurrency_id": "inference", "concurrency_limit": 1}

        def export_indices(selected):
            values = sorted({int(i) for i in (selected or [])})
            if any(i < 1 or i > 28 for i in values):
                raise ValueError("Export channels must be between 1 and 28.")
            return [i - 1 for i in values]

        def export_summary(record, selected):
            indices = export_indices(selected)
            if record is None:
                return "Reconstruct an image first to enable export."
            if not indices:
                return "No channels selected. Tick one or more checkboxes above."
            noun = "channel" if len(indices) == 1 else "channels"
            return f"**{len(indices)} {noun} selected.** Click **Export selected channels**, then download the ZIP."

        def cleared(old):
            cleanup_result(old)
            return {
                overview: None,
                gallery: gr.update(value=[], selected_index=None),
                measurement_image: None,
                mask_image: None,
                spectrum: None,
                mat_file: None,
                zip_file: None,
                export_help: "Reconstruct an image first to enable export.",
                export_button: gr.update(interactive=False),
                state: None,
                status: "Ready. Upload an RGB image to begin.",
            }

        def channel_view(record, channel, gain, pseudocolor, wave):
            channel = int(channel)
            if not 1 <= channel <= 28:
                raise ValueError("Choose a channel between 1 and 28.")
            preview = (
                None
                if record is None
                else band_image(record["cube"], channel - 1, wave, gain, pseudocolor)
            )
            return gr.update(
                value=preview, label=f"Reconstructed HSI · Ch {channel} · {wave[channel - 1]:g} nm"
            )

        def render(record, channel, gain, pseudocolor, labels, selected):
            wave = parse_wavelengths(labels)
            preview = channel_view(record, channel, gain, pseudocolor, wave)
            indices = export_indices(selected)
            if record is None:
                return {
                    overview: preview,
                    gallery: gr.update(value=[], selected_index=None),
                    spectrum: None,
                    mat_file: None,
                    zip_file: None,
                    export_help: export_summary(None, selected),
                    export_button: gr.update(interactive=False),
                }
            mat, _ = export_result(
                record, wave, range(28), gain, pseudocolor, include_channels=False
            )
            return {
                overview: preview,
                gallery: gr.update(
                    value=channel_gallery(record["cube"], range(28), wave, gain, pseudocolor),
                    selected_index=int(channel) - 1,
                ),
                spectrum: spectrum_figure(record, wave),
                mat_file: mat,
                zip_file: None,
                export_help: export_summary(record, selected),
                export_button: gr.update(interactive=bool(indices)),
            }

        def update_view(*args):
            try:
                return render(*args)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        def reconstruct(
            image, key, channel, gain, pseudocolor, labels, selected, old, progress=gr.Progress()
        ):
            record = None
            try:
                parse_wavelengths(labels)
                export_indices(selected)
                if key not in MODELS:
                    raise ValueError(
                        "No model selected. Run python scripts/download_demo_assets.py."
                    )
                progress(0.1, desc="Generating synthetic HSI and CASSI measurement")
                prepared = engine.prepare(image)
                record = engine.reconstruct(prepared, key, progress)
                record["artifact_dir"] = tempfile.mkdtemp(prefix="run-", dir=output_root)
                progress(0.9, desc="Preparing previews and raw HSI download")
                result = render(record, channel, gain, pseudocolor, labels, selected)
                y = prepared["measurement"]
                measurement_preview = np.rint(
                    np.clip(y / max(float(y.max()), 1e-8), 0, 1) * 255
                ).astype(np.uint8)
                h, w, _ = record["cube"].shape
                elapsed = record["synthesis_seconds"] + record["inference_seconds"]
                message = f"**{MODELS[key].label}** · {w} × {h} × 28 bands · {elapsed:.2f} s · Reconstruction complete."
                if record["range_warning"]:
                    gr.Warning(record["range_warning"])
                    message += "\n\n**Output range warning:** " + record["range_warning"]
                result.update(
                    {
                        measurement_image: measurement_preview,
                        mask_image: np.rint(prepared["mask"] * 255).astype(np.uint8),
                        state: record,
                        status: message,
                    }
                )
                cleanup_result(old)
                progress(1, desc="Done")
                return result
            except (ValueError, OSError) as exc:
                cleanup_result(record)
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                cleanup_result(record)
                LOG.exception("Reconstruction failed")
                raise gr.Error(f"Reconstruction failed: {exc}") from exc

        def begin_reconstruction(old):
            result = cleared(old)
            result[status] = "**Reconstructing…**"
            return result

        run.click(
            begin_reconstruction, [state], outputs, api_name=False, show_progress="hidden", **serial
        ).then(
            reconstruct,
            [source, model, band, brightness, color, wavelength_text, export_channels, state],
            outputs,
            api_name="reconstruct",
            show_progress="minimal",
            show_progress_on=[status],
            **serial,
        )
        if example_gallery is not None:

            def select_example(event: gr.SelectData):
                return str(examples[int(event.index)])

            example_gallery.select(
                select_example,
                outputs=source,
                api_name=False,
                show_progress="hidden",
                trigger_mode="always_last",
                **serial,
            )
        source.change(cleared, [state], outputs, api_name=False, show_progress="hidden", **serial)
        clear.click(
            lambda old: {source: None, **cleared(old)},
            [state],
            [source] + outputs,
            api_name=False,
            show_progress="hidden",
            **serial,
        )
        refresh.click(
            lambda: gr.update(choices=engine.available_models()),
            outputs=model,
            api_name=False,
            show_progress="hidden",
        )

        def inspect_thumbnail(record, gain, pseudocolor, labels, event: gr.SelectData):
            if record is None:
                return {overview: gr.skip(), band: gr.skip()}
            try:
                channel = int(event.index) + 1
                wave = parse_wavelengths(labels)
                return {
                    overview: channel_view(record, channel, gain, pseudocolor, wave),
                    band: channel,
                }
            except (ValueError, TypeError) as exc:
                raise gr.Error(str(exc)) from exc

        gallery.select(
            inspect_thumbnail,
            [state, brightness, color, wavelength_text],
            [overview, band],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            **serial,
        )
        view_inputs = [state, band, brightness, color, wavelength_text, export_channels]
        brightness.release(
            update_view, view_inputs, view_outputs, api_name=False, show_progress="hidden", **serial
        )
        color.change(
            update_view, view_inputs, view_outputs, api_name=False, show_progress="hidden", **serial
        )
        apply_settings.click(
            update_view,
            view_inputs,
            view_outputs,
            api_name="update_wavelengths",
            show_progress="hidden",
            **serial,
        )

        def reset_wavelengths(record, channel, gain, pseudocolor, selected):
            return {
                wavelength_text: DEFAULT_WAVELENGTH_TEXT,
                **render(record, channel, gain, pseudocolor, DEFAULT_WAVELENGTH_TEXT, selected),
            }

        reset_settings.click(
            reset_wavelengths,
            [state, band, brightness, color, export_channels],
            [wavelength_text] + view_outputs,
            api_name=False,
            show_progress="hidden",
            **serial,
        )

        def selection_changed(record, selected):
            indices = export_indices(selected)
            return {
                zip_file: None,
                export_help: export_summary(record, selected),
                export_button: gr.update(interactive=record is not None and bool(indices)),
            }

        export_channels.change(
            selection_changed,
            [state, export_channels],
            [zip_file, export_help, export_button],
            api_name=False,
            show_progress="hidden",
            **serial,
        )
        select_all.click(
            lambda: native_channels,
            outputs=export_channels,
            api_name=False,
            show_progress="hidden",
            **serial,
        )
        select_none.click(
            lambda: [], outputs=export_channels, api_name=False, show_progress="hidden", **serial
        )
        select_current.click(
            lambda channel: [int(channel)],
            [band],
            export_channels,
            api_name=False,
            show_progress="hidden",
            **serial,
        )

        def export_selected(record, selected, labels, gain, pseudocolor):
            if record is None:
                raise gr.Error("Reconstruct an image first.")
            try:
                indices = export_indices(selected)
                if not indices:
                    raise ValueError("Tick at least one channel checkbox before exporting.")
                wave = parse_wavelengths(labels)
                _, zipped = export_result(
                    record, wave, indices, gain, pseudocolor, include_cube=False
                )
                names = ", ".join(str(i + 1) for i in indices)
                noun = "channel" if len(indices) == 1 else "channels"
                return (
                    zipped,
                    f"**ZIP ready · {len(indices)} {noun}:** {names}. Click the ZIP file below to download.",
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        export_button.click(
            export_selected,
            [state, export_channels, wavelength_text, brightness, color],
            [zip_file, export_help],
            api_name="export_selected_channels",
            show_progress="minimal",
            show_progress_on=[export_help],
            **serial,
        )

        def select_pixel(record, labels, event: gr.SelectData):
            if record is None:
                return None
            try:
                wave = parse_wavelengths(labels)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            x, y = map(int, event.index)
            h, w, _ = record["cube"].shape
            return spectrum_figure(record, wave, (min(max(x, 0), w - 1), min(max(y, 0), h - 1)))

        overview.select(
            select_pixel,
            [state, wavelength_text],
            spectrum,
            api_name=False,
            show_progress="hidden",
            **serial,
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Model zoo directory; auto-detects simulation/test_code/model_zoo or pretrained_weights",
    )
    parser.add_argument("--mask", type=Path, default=ROOT / "datasets/TSA_simu_data/mask.mat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7876)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    try:
        engine = Reconstructor(args.device, args.weights_dir, args.mask)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Local demo: http://127.0.0.1:{args.port}", flush=True)
    print(
        f"If this is a remote server, run on YOUR computer:\n"
        f"  ssh -N -L {args.port}:127.0.0.1:{args.port} "
        f"{getpass.getuser()}@{socket.gethostname()}",
        flush=True,
    )
    build_demo(engine).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
