import subprocess
from pathlib import Path
import shutil
from PIL import Image, ImageOps

# Optional: raise Pillow's max a bit (do NOT set None unless you trust inputs)
# Your error was at 384M pixels; default limit ~179M. We'll allow up to 600M but still
# keep outputs under our own max_pixels budget via safe DPI clamping below.
Image.MAX_IMAGE_PIXELS = 600_000_000


def _crop_content_bbox(im: Image.Image, pad: int = 20, white_thresh: int = 240) -> Image.Image:
    """
    Tight-crop to content on ALL sides with padding.
    Safe now because we already ensured PDF isn't clipping by using huge paper + auto_widen.
    """
    rgb = im.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    mask = gray.point(lambda p: 255 if p < white_thresh else 0, mode="L")

    bbox = mask.getbbox()
    if not bbox:
        return rgb

    l, t, r, b = bbox
    l = max(l - pad, 0)
    t = max(t - pad, 0)
    r = min(r + pad, rgb.width)
    b = min(b + pad, rgb.height)

    return rgb.crop((l, t, r, b))

def concat_pngs_vertically(png_paths, out_path, page_gap: int = 12):
    pages = []
    max_w = 0
    for p in png_paths:
        im = Image.open(p).convert("RGB")
        im = _crop_content_bbox(im, pad=20)
        pages.append(im)
        max_w = max(max_w, im.width)

    norm = []
    for im in pages:
        if im.width != max_w:
            new_h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, new_h), Image.LANCZOS)
        norm.append(im)

    total_h = sum(im.height for im in norm) + page_gap * (len(norm) - 1)
    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for i, im in enumerate(norm):
        canvas.paste(im, (0, y))
        y += im.height
        if i < len(norm) - 1:
            y += page_gap
    canvas.save(out_path, "PNG")


def wrap_table_with_resizebox(content: str) -> str:
    """Center tabular without resizing (we solve clipping via big paper)."""
    if "\\begin{tabular" not in content:
        return content
    start = content.find("\\begin{tabular")
    end = content.find("\\end{tabular}") + len("\\end{tabular}")
    if start == -1 or end == -1:
        return content
    tabular_env = content[start:end]
    wrapped = "\\noindent\\centering\n" + tabular_env + "\n"
    return content[:start] + wrapped + content[end:]


def has_unicode_chars(text: str) -> bool:
    try:
        text.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def prepare_latex_content(
    original_content: str,
    use_xelatex: bool = False,
    use_preview: bool = False,
    paperwidth_in: float = 40.0,
    paperheight_in: float = 60.0,
    margin_in: float = 0.5,
) -> str:
    """
    Option B: huge paper size so LaTeX doesn't clip wide tables.

    IMPORTANT FIX:
    - Apply settings using \\geometry{...} (overrides prior geometry config)
    - Hard-force PDF MediaBox for pdfTeX using \\pdfpagewidth/\\pdfpageheight
    """
    preview_block = ""
    if use_preview:
        preview_block = (
            "\\usepackage[active,tightpage]{preview}\n"
            "\\PreviewEnvironment{tabular}\n"
            "\\PreviewEnvironment{table}\n"
            "\\PreviewBorder=12pt\n"
        )

    geometry_pkg = "\\usepackage{geometry}\n"
    geometry_apply = (
        f"\\geometry{{paperwidth={paperwidth_in}in,paperheight={paperheight_in}in,margin={margin_in}in}}\n"
    )
    pdfpage_override = (
        f"\\pdfpagewidth={paperwidth_in}in\n"
        f"\\pdfpageheight={paperheight_in}in\n"
    )

    base = (
        geometry_pkg +
        "\\usepackage{graphicx}\n"
        "\\usepackage{adjustbox}\n"
        "\\usepackage{float}\n"
        + preview_block +
        "\\pagestyle{empty}\n"
        "\\sloppy\n"
        "\\setlength{\\emergencystretch}{3em}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\setlength{\\parskip}{0pt}\n"
        "\\setlength{\\floatsep}{0pt}\n"
        "\\setlength{\\textfloatsep}{0pt}\n"
        "\\setlength{\\intextsep}{0pt}\n"
        + geometry_apply +
        (pdfpage_override if not use_xelatex else "")
    )

    if use_xelatex:
        base += "\\usepackage{fontspec}\n"

    if "\\begin{document}" in original_content:
        original_content = original_content.replace("\\begin{document}", base + "\\begin{document}", 1)
    else:
        original_content = base + original_content

    return wrap_table_with_resizebox(original_content)


def _edge_has_content(
    im: Image.Image,
    side: str = "right",
    edge_px: int = 18,
    white_thresh: int = 245,
    min_content_frac: float = 0.002,
) -> bool:
    """
    Detect whether there's non-white content hugging an edge (likely clipping).
    """
    gray = ImageOps.grayscale(im.convert("RGB"))
    w, h = gray.size

    if side == "right":
        box = (max(w - edge_px, 0), 0, w, h)
    elif side == "left":
        box = (0, 0, min(edge_px, w), h)
    else:
        raise ValueError("side must be 'right' or 'left'")

    strip = gray.crop(box)
    mask = strip.point(lambda p: 1 if p < white_thresh else 0, mode="L")
    content_count = sum(mask.getdata())
    total = strip.size[0] * strip.size[1]
    return (content_count / max(total, 1)) >= min_content_frac


# ---------- NEW: DPI safety clamp ----------
def _safe_dpi_for_page(paperwidth_in: float, paperheight_in: float, target_dpi: int, max_pixels: int) -> int:
    """
    Reduce dpi so (paperwidth*dpi)*(paperheight*dpi) <= max_pixels.
    Keeps dpi >= 72.
    """
    import math
    w_in = max(float(paperwidth_in), 1e-6)
    h_in = max(float(paperheight_in), 1e-6)

    # estimated raster dims from pdftoppm
    w_px = w_in * target_dpi
    h_px = h_in * target_dpi
    pix = w_px * h_px

    if pix <= max_pixels:
        return int(target_dpi)

    scale = math.sqrt(max_pixels / float(pix))
    new_dpi = max(72, int(target_dpi * scale))
    return new_dpi
# -----------------------------------------


def render_latex_to_image(
    tex_path: Path,
    dpi: int = 400,
    paperwidth_in: float = 40.0,
    paperheight_in: float = 60.0,
    margin_in: float = 0.5,
    auto_widen: bool = True,
    max_widen_attempts: int = 4,
    widen_factor: float = 1.6,
    # NEW: pixel budget to prevent gigantic pdftoppm outputs & Pillow bomb errors
    max_raster_pixels: int = 160_000_000,
):
    """
    Render LaTeX to PNG with:
      - Option B: huge paper size
      - Auto widen: re-render if content touches left/right edges (clipping indicator)
      - NEW: Auto-lower DPI if paper is huge to avoid >max_raster_pixels images.
    """
    tmp_tex = tex_path.with_name(tex_path.stem + "__render.tex")

    def _compile(use_xelatex: bool, use_preview: bool, pw: float, ph: float):
        original_content = tex_path.read_text(encoding="utf-8")
        modified_content = prepare_latex_content(
            original_content,
            use_xelatex=use_xelatex,
            use_preview=use_preview,
            paperwidth_in=pw,
            paperheight_in=ph,
            margin_in=margin_in,
        )
        tmp_tex.write_text(modified_content, encoding="utf-8")

        if use_xelatex:
            compile_cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", tmp_tex.name] if shutil.which("latexmk") \
                else ["xelatex", "-interaction=nonstopmode", tmp_tex.name]
        else:
            compile_cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tmp_tex.name] if shutil.which("latexmk") \
                else ["pdflatex", "-interaction=nonstopmode", tmp_tex.name]

        subprocess.run(
            compile_cmd,
            cwd=tex_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        tmp_pdf = tmp_tex.with_suffix(".pdf")

        if not tmp_pdf.exists() and use_xelatex:
            tmp_xdv = tmp_tex.with_suffix(".xdv")
            if tmp_xdv.exists():
                subprocess.run(
                    ["xdvipdfmx", tmp_xdv.name],
                    cwd=tex_path.parent,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if tmp_xdv.exists():
                    tmp_xdv.unlink()

        return tmp_pdf

    def _pdf_to_pngs(tmp_pdf: Path, stem_base: str, safe_dpi: int):
        # IMPORTANT: no "-cropbox"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(safe_dpi), tmp_pdf.name, stem_base],
            cwd=tex_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        page_pngs = sorted(
            tex_path.parent.glob(stem_base + "-*.png"),
            key=lambda p: int(p.stem.split("-")[-1])
        )
        return page_pngs

    def _cleanup_render_pages(stem_base: str):
        for p in tex_path.parent.glob(stem_base + "-*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    try:
        original_content = tex_path.read_text(encoding="utf-8")
        use_xelatex = has_unicode_chars(original_content)

        pw = float(paperwidth_in)
        ph = float(paperheight_in)

        final_png = tex_path.with_suffix(".png")

        for attempt in range(max(1, max_widen_attempts if auto_widen else 1)):
            tmp_pdf = _compile(use_xelatex=use_xelatex, use_preview=True, pw=pw, ph=ph)
            if not tmp_pdf.exists():
                tmp_pdf = _compile(use_xelatex=use_xelatex, use_preview=False, pw=pw, ph=ph)

            if not tmp_pdf.exists():
                print(f"[PDF MISSING] for {tex_path.name}")
                return

            # NEW: compute a safe dpi for current page size
            safe_dpi = _safe_dpi_for_page(pw, ph, dpi, max_pixels=max_raster_pixels)

            stem_base = tex_path.stem + f"__render_pages_{attempt}"
            page_pngs = _pdf_to_pngs(tmp_pdf, stem_base, safe_dpi=safe_dpi)

            if not page_pngs:
                print(f"[PNG MISSING] for {tex_path.name}")
                return

            probe = Image.open(page_pngs[0]).convert("RGB")
            right_clip = _edge_has_content(probe, side="right")
            left_clip = _edge_has_content(probe, side="left")

            if auto_widen and (right_clip or left_clip) and attempt < (max_widen_attempts - 1):
                _cleanup_render_pages(stem_base)
                pw *= widen_factor
                ph *= max(1.0, widen_factor * 0.9)
                continue

            if len(page_pngs) == 1:
                cropped = _crop_content_bbox(Image.open(page_pngs[0]), pad=20)
                cropped.save(final_png)
                _cleanup_render_pages(stem_base)
            else:
                concat_pngs_vertically(page_pngs, final_png)
                _cleanup_render_pages(stem_base)

            print(f"[✔] LaTeX rendered: {final_png.name} (paper {pw:.1f}in x {ph:.1f}in, dpi {safe_dpi})")
            break

        # Cleanup aux produced by tmp_tex
        for ext in [".aux", ".log", ".pdf", ".fdb_latexmk", ".fls", ".xdv"]:
            f = tmp_tex.with_suffix(ext)
            if f.exists():
                f.unlink()
        if tmp_tex.exists():
            tmp_tex.unlink()

    except Exception as e:
        print(f"[ERROR] LaTeX rendering failed for {tex_path.name}: {e}")
        if tmp_tex.exists():
            tmp_tex.unlink()