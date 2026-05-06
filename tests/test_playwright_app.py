import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url, timeout=45):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit server did not become ready: {last_error}")


class PlaywrightAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise unittest.SkipTest("Playwright is not installed. Run `pip install playwright` and `python -m playwright install chromium`.") from exc

        cls.sync_playwright = sync_playwright
        cls.port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"

        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(APP_PATH),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(cls.port),
                "--server.headless",
                "true",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            _wait_for_server(cls.base_url)
        except Exception:
            cls.tearDownClass()
            raise

        cls.playwright = cls.sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True)
        except Exception as exc:
            cls.playwright.stop()
            cls.tearDownClass()
            raise unittest.SkipTest("Playwright Chromium is not installed. Run `python -m playwright install chromium`.") from exc

    @classmethod
    def tearDownClass(cls):
        browser = getattr(cls, "browser", None)
        if browser is not None:
            browser.close()
        playwright = getattr(cls, "playwright", None)
        if playwright is not None:
            playwright.stop()
        server = getattr(cls, "server", None)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if server is not None and server.stdout is not None:
            server.stdout.close()

    def _assert_no_streamlit_error(self, page):
        content = page.content()
        self.assertNotIn("This app has encountered an error", content)
        self.assertNotIn("DuplicateError", content)
        self.assertNotIn("Traceback", content)

    def test_dashboard_pages_load_without_app_error(self):
        page = self.browser.new_page(viewport={"width": 1440, "height": 1100})
        try:
            page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
            page.get_by_text("Wastewater Odor Performance Brief").wait_for(timeout=60000)
            self._assert_no_streamlit_error(page)

            for page_name in [
                "Operations Review",
                "Chemistry & Dosing",
                "Research Progress",
                "Performance & Coverage",
                "Diagnostics & Data",
                "Executive Brief",
            ]:
                page.get_by_text(page_name, exact=True).first.click()
                page.wait_for_timeout(1200)
                self._assert_no_streamlit_error(page)
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
