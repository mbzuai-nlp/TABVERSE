import base64
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import markdown2
from PIL import Image, ImageChops


def _crop_content_bbox(im: Image.Image, pad: int = 12) -> Image.Image:
    """Crop white margins from image, keeping a padding."""
    bg = Image.new(im.mode, im.size, "white")
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    l = max(l - pad, 0)
    t = max(t - pad, 0)
    r = min(r + pad, im.width)
    b = min(b + pad, im.height)
    return im.crop((l, t, r, b))


def convert_md_to_html(md_path: Path, html_output_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown2.markdown(
        md_text,
        extras=["tables", "fenced-code-blocks", "strike", "task_list"],
    )
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.4.0/github-markdown.min.css">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      overflow: visible;
    }}
    .markdown-body {{
      box-sizing: border-box;
      width: fit-content;
      padding: 40px;
      font-size: 22px;
    }}
    table {{
      border-collapse: collapse;
    }}
    td, th {{
      border: 1px solid #ccc;
      padding: 8px;
    }}
  </style>
</head>
<body>
<article class="markdown-body">
{html_body}
</article>
</body>
</html>"""
    html_output_path.write_text(full_html, encoding="utf-8")


def render_html_to_png(html_path: Path, png_path: Path) -> None:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(f"file://{html_path.resolve()}")
        time.sleep(0.8)

        width = driver.execute_script("""
            return Math.max(
                document.body.scrollWidth,
                document.documentElement.scrollWidth
            );
        """)
        height = driver.execute_script("""
            return Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight
            );
        """)

        driver.set_window_size(width, height)

        png_b64 = driver.execute_cdp_cmd(
            "Page.captureScreenshot",
            {
                "format": "png",
                "captureBeyondViewport": True,
                "fromSurface": True
            },
        )["data"]

        # Save and crop
        tmp_png = png_path.with_name(png_path.stem + "_raw.png")
        tmp_png.write_bytes(base64.b64decode(png_b64))

        im = Image.open(tmp_png).convert("RGB")
        cropped = _crop_content_bbox(im, pad=12)
        cropped.save(png_path)
        tmp_png.unlink()

        print(f"Screenshot saved: {png_path}")
    finally:
        driver.quit()


def render_markdown_to_image(md_path: Path):
    html_path = md_path.with_suffix(".temp.html")
    png_path = md_path.with_suffix(".png")
    convert_md_to_html(md_path, html_path)
    render_html_to_png(html_path, png_path)
    html_path.unlink(missing_ok=True)