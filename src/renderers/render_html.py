from pathlib import Path
from playwright.sync_api import sync_playwright

def render_html_to_image(html_path: Path, output_path: Path):
    # Read HTML content (only contains table)
    html_content = html_path.read_text(encoding='utf-8')
    
    # Write to a temporary file
    temp_path = html_path.with_suffix('.tmp.html')
    temp_path.write_text(html_content, encoding='utf-8')
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{temp_path.resolve()}")

        # Inject custom CSS for zoom, padding, font size
        page.add_style_tag(content="""
            body {
                font-size: 25px !important;
                padding: 20px;
                background: white;
            }
            table {
                width: 1200px !important;
                font-size: 20px !important;
                transform: scale(1.2);
                transform-origin: top left;
                margin-top: 20px;
            }
            td, th {
                padding: 12px;
                border: 1px solid #aaa;
            }
        """)

        # Wait to allow layout to settle
        page.wait_for_timeout(1000)

        # Resize viewport to content size with minimum dimensions
        dimensions = page.evaluate("""() => {
            const body = document.body;
            const html = document.documentElement;
            return {
                width: Math.max(body.scrollWidth, html.scrollWidth, 800),
                height: Math.max(body.scrollHeight, html.scrollHeight, 600)
            }
        }""")
        page.set_viewport_size(dimensions)

        # Take screenshot
        page.screenshot(path=output_path.as_posix(), full_page=True)
        print(f"[✔] HTML rendered: {output_path.name}")

        browser.close()
        
        # Clean up temp file
        temp_path.unlink()