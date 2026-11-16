"""Verify compact preview, clickable bands, explicit export selection, and progress timing."""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7876")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/browser-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(error.stack))
        page.goto(args.url, wait_until="domcontentloaded")
        page.get_by_role("button", name="Reconstruct", exact=True).wait_for()
        expect(page).to_have_title("MST · CVPR 2022")
        expect(page.locator("#hero h1")).to_have_text("MST · CVPR 2022")
        for url in [
            "https://github.com/caiyuanhao1998/MST",
            "https://huggingface.co/CaiYuanhao/MST",
            "https://huggingface.co/spaces/CaiYuanhao/Retinexformer",
        ]:
            expect(page.locator(f'a[href="{url}"]')).to_have_count(0)
        expect(page.get_by_text("10 Official SCI RGB examples", exact=True)).to_have_count(1)
        expect(page.get_by_role("button", name="Refresh Weights", exact=True)).to_have_count(1)
        expect(page.get_by_role("button", name="Refresh", exact=True)).to_have_count(0)
        expect(page.get_by_text("How RGB becomes a measurement", exact=True)).to_have_count(0)
        expect(page.get_by_text("Pages:", exact=False)).to_have_count(0)
        overview = page.locator("#reconstruction-preview")
        page.evaluate("""() => {
            window.resultTiming = null;
            const visible = element => {
                if (!element) return false;
                const rect = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                return rect.width > 0 && rect.height > 0 &&
                       style.display !== 'none' && style.visibility !== 'hidden';
            };
            const sample = () => {
                const timing = window.resultTiming;
                if (timing?.active) {
                    const status = document.querySelector('#run-status');
                    const text = status?.textContent || '';
                    const progress = [...(status?.querySelectorAll('.progress-bar') || [])]
                        .some(bar => {
                            const wrap = bar.closest('.wrap');
                            return wrap && !wrap.classList.contains('hide') && visible(bar.parentElement);
                        });
                    const processing = text.includes('Reconstructing') || progress;
                    if (processing) timing.started = true;
                    if (timing.started) {
                        const images = [...document.querySelectorAll(
                            '#measurement-image img, #mask-image img, #reconstruction-preview img, ' +
                            '#channel-image img, #spectrum-plot img, #spectral-gallery img')]
                            .filter(image => image.complete && image.naturalWidth > 0 && visible(image));
                        if (processing) {
                            timing.processing_frames++;
                            timing.images_while_processing = Math.max(timing.images_while_processing, images.length);
                        }
                        if (progress) {
                            timing.progress_frames++;
                            timing.images_while_progress = Math.max(timing.images_while_progress, images.length);
                        }
                        if (text.includes('Reconstruction complete.') && !progress) {
                            timing.completed = true;
                        }
                    }
                }
                requestAnimationFrame(sample);
            };
            requestAnimationFrame(sample);
        }""")

        def reconstruct_with_timing_check():
            page.evaluate("""() => {
                window.resultTiming = {active:true, started:false, completed:false,
                    processing_frames:0, progress_frames:0,
                    images_while_processing:0, images_while_progress:0};
            }""")
            page.get_by_role("button", name="Reconstruct", exact=True).click()
            expect(page.locator("#run-status")).to_contain_text(
                "Reconstruction complete.", timeout=120000
            )
            expect(page.locator("#spectral-gallery .thumbnail-lg")).to_have_count(28)
            page.wait_for_function(
                """() => {
                return window.resultTiming.completed &&
                    ['#measurement-image img','#mask-image img','#reconstruction-preview img']
                    .every(selector => {
                        const image = document.querySelector(selector);
                        return image?.complete && image.naturalWidth > 0;
                    });
            }""",
                timeout=30000,
            )
            timing = page.evaluate("""() => {
                window.resultTiming.active=false;
                return {...window.resultTiming};
            }""")
            assert timing["processing_frames"] > 0, timing
            assert timing["progress_frames"] > 0, timing
            assert timing["images_while_processing"] == 0, timing
            assert timing["images_while_progress"] == 0, timing
            return timing

        examples = page.locator("#sci-examples")
        expect(examples.locator(".thumbnail-lg")).to_have_count(10)
        example_geometry = examples.evaluate("""gallery => {
            const wrap=gallery.querySelector('.grid-wrap');
            const items=[...gallery.querySelectorAll('.thumbnail-lg')];
            return {rows:new Set(items.map(e=>Math.round(e.getBoundingClientRect().top))).size,
                    horizontalScroll:wrap.scrollWidth>wrap.clientWidth};
        }""")
        assert example_geometry["rows"] == 1 and example_geometry["horizontalScroll"], (
            example_geometry
        )
        # The last scene is reachable without pagination and loads the actual RGB pixels.
        examples.get_by_role("button", name="Thumbnail 10 of 10", exact=True).click()
        image = page.locator("#rgb-input img")
        image.wait_for(state="visible")
        last_path = ROOT / "demo/examples/simulation/scene_10.png"
        with Image.open(last_path) as original:
            original_rgb = np.asarray(original.convert("RGB"))
        page.wait_for_function(
            "size => {const i=document.querySelector('#rgb-input img'); return i?.complete && i.naturalWidth===size[0] && i.naturalHeight===size[1]}",
            arg=[original_rgb.shape[1], original_rgb.shape[0]],
        )
        response = page.request.get(image.evaluate("e => e.currentSrc || e.src"))
        assert response.status == 200
        np.testing.assert_array_equal(
            np.asarray(Image.open(io.BytesIO(response.body())).convert("RGB")), original_rgb
        )
        # Use the fourth official scene (297 px), larger than the 256 px model output.
        examples.get_by_role("button", name="Thumbnail 4 of 10", exact=True).click()
        page.wait_for_function("document.querySelector('#rgb-input img')?.naturalWidth === 297")
        first_timing = reconstruct_with_timing_check()
        expect(overview).to_contain_text("Reconstructed HSI · Ch 22 · 584.5 nm")
        repeat_timing = reconstruct_with_timing_check()
        print(
            "PASS: first and repeated runs withhold all result images during progress", flush=True
        )
        expect(page.locator("#channel-image")).to_have_count(0)
        expect(page.locator("#inspect-channel")).to_have_count(0)
        for selector in ["#measurement-image img", "#mask-image img", "#spectrum-plot img"]:
            expect(page.locator(selector)).to_be_visible()
        diagnostics = page.evaluate("""() => {
            const left=document.querySelector('#measurement-panel').getBoundingClientRect();
            const right=document.querySelector('#spectrum-panel').getBoundingClientRect();
            const upper=document.querySelector('#comparison-row').getBoundingClientRect();
            return {measurementOnLeft:left.left<right.left,
                    sameRow:Math.abs(left.top-right.top)<1,
                    belowPreviews:Math.min(left.top,right.top)>=upper.bottom};
        }""")
        assert all(diagnostics.values()), diagnostics

        geometry = page.evaluate("""() => {
            const gallery = document.querySelector('#spectral-gallery');
            const preview = document.querySelector('#reconstruction-preview').getBoundingClientRect();
            const items = [...gallery.querySelectorAll('.thumbnail-lg')].map(e=>e.getBoundingClientRect());
            const wrap = gallery.querySelector('.grid-wrap');
            const strip = gallery.getBoundingClientRect();
            return {pageHeight:document.documentElement.scrollHeight,
                    galleryInResultColumn:document.querySelector('#result-column').contains(gallery),
                    stripHeight:strip.height, stripBelowPreview:strip.top>=preview.bottom,
                    thumbnailRows:new Set(items.map(r=>Math.round(r.top))).size,
                    horizontalScroll:wrap.scrollWidth>wrap.clientWidth};
        }""")
        assert geometry["galleryInResultColumn"] and geometry["stripBelowPreview"], geometry
        assert geometry["thumbnailRows"] == 1 and geometry["horizontalScroll"], geometry
        assert geometry["stripHeight"] <= 140, geometry
        for selector, text in [
            ("#model-dropdown", "Reconstruction model"),
            ("#display-brightness", "Preview brightness"),
        ]:
            expect(page.locator(selector).get_by_text(text, exact=True)).to_be_visible()
        alignment = page.evaluate("""() => {
            const pairs=[['rgb-input','reconstruction-preview'],['sci-examples','spectral-gallery'],
                         ['model-panel','preview-panel'],['run-status','wavelength-settings'],
                         ['measurement-panel','spectrum-panel']];
            const rect=id=>document.getElementById(id).getBoundingClientRect();
            const borders=pairs.flat().map(id=>{const s=getComputedStyle(document.getElementById(id));
                return {id,width:s.borderTopWidth,color:s.borderTopColor,radius:s.borderRadius};});
            const refresh=document.getElementById('refresh-weights');
            const action=rect('reconstruct'),clear=rect('clear-images'),exportPanel=rect('export-panel');
            return {pairs:pairs.map(([a,b])=>{const x=rect(a),y=rect(b);return {
                        left:a,right:b,topDelta:Math.abs(x.top-y.top),
                        bottomDelta:Math.abs(x.bottom-y.bottom),widthDelta:Math.abs(x.width-y.width)};}),
                    borders,refreshOneLine:getComputedStyle(refresh).whiteSpace==='nowrap',
                    actionTopDelta:Math.max(Math.abs(action.top-clear.top),Math.abs(action.top-exportPanel.top)),
                    actionHeightDelta:Math.max(Math.abs(action.height-clear.height),Math.abs(action.height-exportPanel.height))};
        }""")
        for pair in alignment["pairs"]:
            assert max(pair["topDelta"], pair["bottomDelta"], pair["widthDelta"]) <= 1, pair
        assert all(border["width"] == "1px" for border in alignment["borders"]), alignment
        assert len({border["color"] for border in alignment["borders"]}) == 1, alignment
        assert alignment["refreshOneLine"], alignment
        assert alignment["actionTopDelta"] <= 1 and alignment["actionHeightDelta"] <= 1, alignment
        assert geometry["pageHeight"] < 1600, geometry
        baseline = ROOT / ".local/desktop-before-compact-layout.png"
        previous_height = Image.open(baseline).height if baseline.exists() else None
        if previous_height and previous_height > 1600:
            assert geometry["pageHeight"] < previous_height * 0.7, geometry

        def equal_preview_dimensions():
            metrics = page.evaluate("""() => {
                const measure = selector => {
                    const image=document.querySelector(selector),rect=image.getBoundingClientRect();
                    const panel=image.closest('#rgb-input, #reconstruction-preview').getBoundingClientRect();
                    const scale=Math.min(rect.width/image.naturalWidth,rect.height/image.naturalHeight);
                    return {frameWidth:panel.width,frameHeight:panel.height,fit:getComputedStyle(image).objectFit,
                            width:scale*image.naturalWidth,height:scale*image.naturalHeight,
                            nativeWidth:image.naturalWidth,nativeHeight:image.naturalHeight};
                };
                return {input:measure('#rgb-input img'),output:measure('#reconstruction-preview img')};
            }""")
            a, b = metrics["input"], metrics["output"]
            assert a["fit"] == b["fit"] == "contain", metrics
            for dimension in ("frameWidth", "frameHeight", "width", "height"):
                assert abs(a[dimension] - b[dimension]) <= 1, metrics
            return metrics

        preview_dimensions = equal_preview_dimensions()
        assert preview_dimensions["input"]["nativeWidth"] == 297
        assert preview_dimensions["output"]["nativeWidth"] == 256
        yellow = page.evaluate("""() => {
            const image=document.querySelector('#reconstruction-preview img');
            const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
            const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
            const p=ctx.getImageData(0,0,canvas.width,canvas.height).data;
            let r=0,g=0,b=0;for(let i=0;i<p.length;i+=4){r+=p[i];g+=p[i+1];b+=p[i+2];}
            return g>0 && r/g>0.9 && r/g<1.12 && b/g<0.01;
        }""")
        assert yellow, "Default channel must remain yellow"
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        page.locator("#comparison-row").screenshot(path=str(args.output / "comparison.png"))

        # The strip changes only the upper-right preview; it does not open a gallery modal.
        message = page.locator("#run-status").inner_text()
        page.locator("#spectral-gallery").get_by_role(
            "button", name="Thumbnail 14 of 28", exact=True
        ).click()
        expect(overview).to_contain_text("Reconstructed HSI · Ch 14 · 522.5 nm", timeout=30000)
        expect(page.locator("#spectral-gallery .preview")).to_have_count(0)
        expect(page.locator("#run-status")).to_have_text(message)
        plot = page.locator("#spectrum-plot img")
        plot.wait_for(state="visible")
        old_plot = plot.get_attribute("src")
        overview.locator("img").click()
        page.wait_for_function(
            "previous => document.querySelector('#spectrum-plot img')?.src !== previous",
            arg=old_plot,
            timeout=30000,
        )

        print("PASS: compact layout, thumbnail preview, and pixel spectrum", flush=True)
        panel = page.locator("#export-panel")
        panel.get_by_text("Export selected channels", exact=True).first.click()
        selection = page.locator("#export-channel-selection")
        expect(selection.locator("input[type=checkbox]")).to_have_count(28)
        expect(selection.locator("input[type=checkbox]:checked")).to_have_count(28)
        panel.get_by_role("button", name="Current channel only", exact=True).click()
        expect(selection.locator("input[type=checkbox]:checked")).to_have_count(1)
        expect(selection.get_by_label("Ch 14", exact=True)).to_be_checked()
        expect(page.locator("#export-help")).to_contain_text("1 channel selected", timeout=30000)
        page.locator("#export-channels-button").click()
        expect(page.locator("#export-help")).to_contain_text("ZIP ready", timeout=30000)

        def download(selector, target):
            with page.expect_download() as event:
                page.locator(selector).first.click()
            event.value.save_as(str(target))

        single_zip = args.output / "current-channel.zip"
        download('#channel-zip a[href*="spectral_channels.zip"]', single_zip)
        with zipfile.ZipFile(single_zip) as archive:
            files = [name for name in archive.namelist() if name.endswith(".png")]
            assert len(files) == 2, files
            assert json.loads(archive.read("metadata.json"))["export_channels_1based"] == [14]
            exported = np.asarray(
                Image.open(io.BytesIO(archive.read("display/ch_14_522.5nm.png"))).convert("RGB")
            )
        response = page.request.get(overview.locator("img").evaluate("(e)=>e.currentSrc || e.src"))
        assert response.status == 200
        shown = np.asarray(Image.open(io.BytesIO(response.body())).convert("RGB"))
        np.testing.assert_array_equal(shown, exported)

        print(
            "PASS: current-channel ZIP contains the exact preview and selected metadata", flush=True
        )

        # Explicit export choices must not filter the thumbnail strip or alter the current preview.
        panel.get_by_role("button", name="Select none", exact=True).click()
        expect(selection.locator("input[type=checkbox]:checked")).to_have_count(0)
        expect(page.locator("#export-channels-button")).to_be_disabled()
        expect(page.locator('#channel-zip a[href*="spectral_channels.zip"]')).to_have_count(0)
        for channel in [1, 7, 14, 28]:
            selection.get_by_label(f"Ch {channel:02d}", exact=True).check()
        expect(page.locator("#export-help")).to_contain_text("4 channels selected", timeout=30000)
        expect(page.locator("#spectral-gallery .thumbnail-lg")).to_have_count(28)
        expect(overview).to_contain_text("Ch 14 · 522.5 nm")

        page.get_by_text("Wavelength labels (advanced)", exact=True).click()
        labels = page.locator("#wavelength-labels textarea")
        native_text = labels.input_value()
        labels.fill(", ".join(str(float(x.strip()) + 1) for x in native_text.split(",")))
        page.get_by_role("button", name="Apply wavelength labels", exact=True).click()
        expect(overview).to_contain_text("Ch 14 · 523.5 nm", timeout=30000)
        expect(page.locator("#spectral-gallery .thumbnail-lg")).to_have_count(28)
        page.locator("#export-channels-button").click()
        expect(page.locator("#export-help")).to_contain_text("ZIP ready", timeout=30000)
        archive_path = args.output / "channels.zip"
        download('#channel-zip a[href*="spectral_channels.zip"]', archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            assert len([name for name in archive.namelist() if name.endswith(".png")]) == 8
            meta = json.loads(archive.read("metadata.json"))
            assert meta["export_channels_1based"] == [1, 7, 14, 28]
            assert meta["display_wavelengths_nm"][0] == 454.5
            assert meta["native_wavelengths_nm"][0] == 453.5

        mat_path = args.output / "reconstruction.mat"
        download('#full-cube-mat a[href*="reconstruction.mat"]', mat_path)
        data = sio.loadmat(mat_path)
        assert data["cube"].shape[-1] == 28
        assert data["measurement"].shape == (256, 310)
        assert data["wavelengths_nm"].ravel()[0] == 453.5
        assert data["display_wavelengths_nm"].ravel()[0] == 454.5
        assert json.loads(str(data["metadata_json"][0]))["export_channels_1based"] == list(
            range(1, 29)
        )
        original_link = page.locator(
            '#full-cube-mat a[href*="reconstruction.mat"]'
        ).first.get_attribute("href")
        gain = page.locator("#display-brightness input[type=number]")
        gain.fill("2")
        gain.press("Tab")
        expect(
            page.locator('#full-cube-mat a[href*="reconstruction.mat"]').first
        ).not_to_have_attribute("href", original_link, timeout=30000)
        expect(page.locator('#channel-zip a[href*="spectral_channels.zip"]')).to_have_count(0)
        bright = args.output / "brightness_2.mat"
        download('#full-cube-mat a[href*="reconstruction.mat"]', bright)
        np.testing.assert_array_equal(sio.loadmat(bright)["cube"], data["cube"])
        page.screenshot(path=str(args.output / "custom-channels.png"), full_page=True)
        page.get_by_role("button", name="Reset repository defaults", exact=True).click()
        expect(labels).to_have_value(native_text, timeout=30000)
        expect(overview).to_contain_text("Ch 14 · 522.5 nm")
        expect(selection.locator("input[type=checkbox]:checked")).to_have_count(4)
        page.get_by_text("Wavelength labels (advanced)", exact=True).click()
        panel.get_by_text("Export selected channels", exact=True).first.click()

        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        overflow = page.evaluate("document.documentElement.scrollWidth>innerWidth")
        assert not overflow, "Mobile page overflows horizontally"
        equal_preview_dimensions()
        page.locator("#clear-images").click()
        expect(page.locator("#run-status")).to_contain_text(
            "Ready. Upload an RGB image to begin.", timeout=30000
        )
        expect(page.locator("#spectral-gallery .thumbnail-lg")).to_have_count(0)
        expect(overview.locator("img")).to_have_count(0)
        assert not errors, errors
        result = {
            "url": args.url,
            "passed": True,
            "browser_errors": errors,
            "results_visible_only_after_completion": True,
            "no_images_while_progress_active": True,
            "initial_run_timing": first_timing,
            "repeat_run_timing": repeat_timing,
            "compact_page_height": geometry["pageHeight"],
            "previous_page_height": previous_height,
            "one_horizontal_thumbnail_row": True,
            "thumbnail_click_updates_upper_preview": True,
            "no_duplicate_large_preview": True,
            "export_selection_separate_from_preview": True,
            "current_channel_export_verified": True,
            "checkbox_selected_channels": [1, 7, 14, 28],
            "download_pngs": 8,
            "full_mat_keeps_all_28_bands": True,
            "native_wavelengths_preserved": True,
            "brightness_preserves_raw_hsi": True,
            "pixel_spectrum": True,
            "mobile_horizontal_overflow": overflow,
            "reset_and_clear": True,
            "title_and_footer_updated": True,
            "official_sci_rgb_examples": 10,
            "example_gallery_single_scroll_row": example_geometry,
            "example_pagination_removed": True,
            "last_example_click_loads_original_pixels": True,
            "refresh_weights_label": True,
            "rgb_formula_section_removed": True,
            "measurement_mask_spectrum_default_expanded": True,
            "diagnostics_layout": diagnostics,
            "panel_alignment": alignment,
            "default_preview_channel": 22,
            "default_wavelength_nm": 584.5,
            "default_preview_is_yellow": yellow,
            "matched_preview_dimensions": preview_dimensions,
        }
        (ROOT / "docs/browser_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
