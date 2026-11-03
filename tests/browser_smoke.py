"""Run explicitly: python tests/browser_smoke.py (requires Chromium)."""
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import urllib.request

import cv2
import numpy as np
import pymupdf
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    image = np.full((400, 500), 255, np.uint8)
    cv2.rectangle(image, (60, 60), (440, 340), 0, 3)
    with pymupdf.open() as pdf:
        for _ in range(2):
            page = pdf.new_page(width=250, height=200)
            page.insert_image(page.rect, stream=cv2.imencode('.png', image)[1].tobytes())
        data = pdf.tobytes()
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                               '--host', '127.0.0.1', '--port', '8765'], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for attempt in range(100):
            try:
                urllib.request.urlopen('http://127.0.0.1:8765/', timeout=1).close()
                break
            except OSError:
                if server.poll() is not None or attempt == 99:
                    raise RuntimeError('Test server failed to start')
                time.sleep(0.1)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.environ.get('CHROMIUM_PATH') or shutil.which('chromium'),
                args=['--no-sandbox'])
            page = browser.new_page(accept_downloads=True)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto('http://127.0.0.1:8765/')
            page.locator('#file').set_input_files({'name': 'scan.pdf', 'mimeType': 'application/pdf', 'buffer': data})
            expect(page.locator('#process')).to_be_enabled()
            page.locator('#page').select_option('2')
            expect(page.locator('#original')).to_be_visible()
            page.wait_for_function("document.getElementById('original').naturalWidth > 0")
            page.locator('#process').click()
            expect(page.locator('#status')).to_contain_text('変換しました', timeout=20000)
            page.wait_for_function("document.getElementById('corrected').naturalWidth > 0")
            for selector, prefix in [('#pdf', b'%PDF'), ('#svg', b'<svg')]:
                with page.expect_download() as download_info:
                    page.locator(selector).click()
                download = download_info.value
                assert Path(download.path()).read_bytes().startswith(prefix)
            expect(page.locator('#review')).to_be_visible()
            page.wait_for_function("document.getElementById('reference').naturalWidth > 0")
            page.locator('#opacity').fill('25')
            page.locator('#opacity').dispatch_event('input')
            expect(page.locator('#overlay')).to_have_css('opacity', '0.25')
            page.locator('#review-mode').select_option('difference')
            expect(page.locator('#difference')).to_be_visible()
            page.wait_for_function("document.getElementById('difference').naturalWidth > 0")
            page.locator('#review-mode').select_option('uncertain')
            expect(page.locator('#uncertain')).to_be_visible()
            page.wait_for_function("document.getElementById('uncertain').naturalWidth > 0")
            with page.expect_download() as preserved_info:
                page.locator('#preserved').click()
            assert Path(preserved_info.value.path()).read_bytes().startswith(b'%PDF')
            page.locator('#noise-strength').fill('0.7')
            page.locator('#noise-strength').dispatch_event('change')
            expect(page.locator('#svg')).not_to_have_attribute('href')
            expect(page.locator('#review')).to_be_hidden()
            # Retain the existing workflow in the legacy mode too.
            page.locator('#mode').select_option('legacy')
            page.locator('#process').click()
            expect(page.locator('#status')).to_contain_text('変換しました', timeout=20000)
            expect(page.locator('#review')).to_be_hidden()
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#close').click()
            expect(page.locator('#status')).to_contain_text('図面を削除しました')
            expect(page.locator('#process')).to_be_disabled()
            # Dense architectural-like page: real background job and progress UI.
            large = np.full((1600, 2400), 255, np.uint8)
            for y in range(20, 1580, 35):
                cv2.line(large, (15, y), (2385, y), 0, 5)
            with pymupdf.open() as large_pdf:
                p = large_pdf.new_page(width=1200, height=800)
                p.insert_image(p.rect, stream=cv2.imencode('.png', large)[1].tobytes())
                large_data = large_pdf.tobytes()
            page.locator('#file').set_input_files({'name':'large.pdf', 'mimeType':'application/pdf', 'buffer':large_data})
            expect(page.locator('#process')).to_be_enabled()
            page.locator('#mode').select_option('cad_clean')
            page.locator('#deskew').uncheck()
            page.locator('#noise-strength').fill('0')
            page.locator('#process').click()
            expect(page.locator('#progress')).to_be_visible()
            expect(page.locator('#status')).to_contain_text('小領域')
            expect(page.locator('#status')).to_contain_text('変換しました', timeout=60000)
            expect(page.locator('#progress')).to_be_hidden()
            expect(page.locator('#svg')).to_have_attribute('aria-disabled', 'false')
            # Transport/server rejection must be displayed in Japanese, no stale link.
            page.route('**/jobs', lambda route: route.fulfill(status=503,
                content_type='application/json', body='{"detail":"処理用メモリが不足しました。"}'))
            page.locator('#process').click()
            expect(page.locator('#status')).to_contain_text('処理用メモリが不足')
            expect(page.locator('#svg')).not_to_have_attribute('href')
            expect(page.locator('#preserved')).to_have_attribute('aria-disabled', 'false')
            expect(page.locator('#progress')).to_be_hidden()
            page.locator('#close').click()
            expect(page.locator('#process')).to_be_disabled()
            assert not errors, errors
            browser.close()
        print('Browser smoke passed: dense-plan progress, Japanese failure,  upload, page switch, process, comparison, PDF/SVG downloads, mobile layout, deletion')
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == '__main__':
    main()
