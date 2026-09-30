"""Adapter contract tests.

Two halves:

  - ``test_abc_*`` — verify that the abstract contract refuses partial
    implementations and that every concrete adapter in this module
    exposes every method on the ABC. These tests don't launch a
    browser; they're pure interface checks.

  - ``test_playwright_*`` — drive the Playwright adapter against a
    pytest-httpserver fixture (no real URLs, no mocks) to verify the
    page-level operations actually do what the ABC promises.

The CDP adapter needs a running ``BrowserSession`` against a real
Chromium subprocess to exercise. Those tests live alongside the rest
of the CDP-flavoured suite and aren't duplicated here — the ABC-level
checks below already lock the surface.
"""

from __future__ import annotations

import inspect

import pytest

from browser_use.browser.adapter import (
	BrowserAdapter,
	CdpBrowserAdapter,
	PlaywrightBrowserAdapter,
)

# ── ABC contract ────────────────────────────────────────────────────────────


def test_browser_adapter_is_abstract() -> None:
	"""You can't instantiate the bare ABC — guards against accidental
	use of the placeholder contract."""
	with pytest.raises(TypeError):
		BrowserAdapter()  # type: ignore[abstract]


def test_concrete_adapters_implement_every_abstract_method() -> None:
	"""Every method declared abstract on BrowserAdapter must be
	concretely implemented on every shipped adapter. This is what
	makes the abstraction load-bearing — consumers should never need
	to check the adapter's type before calling a method."""
	abstract_methods = {
		name for name, method in BrowserAdapter.__dict__.items() if getattr(method, '__isabstractmethod__', False)
	}
	assert abstract_methods, 'BrowserAdapter should have abstract methods'

	for adapter_cls in (CdpBrowserAdapter, PlaywrightBrowserAdapter):
		assert not adapter_cls.__abstractmethods__, (
			f'{adapter_cls.__name__} is missing concrete impls for {adapter_cls.__abstractmethods__}'
		)
		# Belt-and-braces: every abstract name must resolve to a real
		# coroutine function on the subclass.
		for name in abstract_methods:
			fn = getattr(adapter_cls, name, None)
			assert fn is not None, f'{adapter_cls.__name__} is missing {name}'
			assert inspect.iscoroutinefunction(fn), f'{adapter_cls.__name__}.{name} should be `async def`'


def test_concrete_adapter_signatures_match_abc() -> None:
	"""Parameter names and types stay aligned across implementations.
	Consumers pass kwargs (timeout_ms=…) — if an adapter renames a
	parameter, those calls silently break. Lock the signatures here."""
	for name, abc_method in BrowserAdapter.__dict__.items():
		if not getattr(abc_method, '__isabstractmethod__', False):
			continue
		abc_sig = inspect.signature(abc_method)
		for adapter_cls in (CdpBrowserAdapter, PlaywrightBrowserAdapter):
			impl_sig = inspect.signature(getattr(adapter_cls, name))
			assert list(impl_sig.parameters) == list(abc_sig.parameters), (
				f'{adapter_cls.__name__}.{name} param order diverges from ABC: '
				f'{list(impl_sig.parameters)} vs {list(abc_sig.parameters)}'
			)


# ── Playwright adapter end-to-end ───────────────────────────────────────────
#
# These run only when Playwright is installed AND a browser binary is
# available. Camoufox's Firefox is tried first — it is the backend this
# adapter actually exists to drive, and `camoufox fetch` puts it on disk
# for the Firefox engine anyway. Chromium is the fallback so the suite
# still exercises the adapter on a CI image that only installed Chromium.
# Skip cleanly when neither is present.


@pytest.fixture
async def playwright_page():
	"""A real Playwright page: Camoufox Firefox if fetched, else Chromium.

	Async (the suite runs asyncio_mode=auto) so the browser lives on the same
	event loop as the test. A sync fixture driving run_until_complete here
	deadlocks against the loop pytest-asyncio already owns.
	"""
	try:
		from playwright.async_api import async_playwright
	except ImportError:
		pytest.skip('playwright not installed')

	pw = await async_playwright().start()

	async def _launch_camoufox():
		# AsyncNewBrowser applies camoufox's own launch_options (CAMOU_CONFIG,
		# addons, the no_viewport workaround), so this is the same bring-up the
		# Firefox engine uses — not a bare binary launch.
		from camoufox import AsyncNewBrowser

		return await AsyncNewBrowser(pw, headless=True)

	async def _launch_chromium():
		return await pw.chromium.launch(headless=True)

	failures = []
	browser = None
	for launch in (_launch_camoufox, _launch_chromium):
		try:
			browser = await launch()
			break
		except Exception as e:  # noqa: BLE001 — any launch failure means "try the next backend"
			failures.append(f'{launch.__name__.removeprefix("_launch_")}: {e}')
	if browser is None:
		await pw.stop()
		pytest.skip('no Playwright browser available — ' + '; '.join(failures))

	context = await browser.new_context()
	page = await context.new_page()
	try:
		yield page
	finally:
		for step in (page.close, context.close, browser.close, pw.stop):
			try:
				await step()
			except Exception:  # noqa: BLE001 — teardown is best-effort
				pass


async def test_playwright_adapter_goto_and_content(playwright_page, httpserver) -> None:
	httpserver.expect_request('/index').respond_with_data(
		'<html><head><title>OK</title></head><body><h1 class="g">Hi</h1></body></html>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	result = await adapter.goto(httpserver.url_for('/index'))
	assert result['status'] == 200
	html = await adapter.content()
	assert 'Hi' in html
	assert await adapter.title() == 'OK'


async def test_playwright_adapter_locator(playwright_page, httpserver) -> None:
	httpserver.expect_request('/list').respond_with_data(
		'<ul><li class="x">a</li><li class="x">b</li><li class="x">c</li></ul>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/list'))
	assert await adapter.locator_count('li.x') == 3
	assert await adapter.locator_inner_text('li.x') == 'a'
	assert await adapter.locator_get_attribute('li.x', 'class') == 'x'
	assert await adapter.locator_is_visible('li.x') is True
	assert await adapter.locator_count('li.missing') == 0
	assert await adapter.locator_is_visible('li.missing') is False


async def test_playwright_adapter_evaluate(playwright_page, httpserver) -> None:
	httpserver.expect_request('/blank').respond_with_data('<html><body></body></html>', content_type='text/html')
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/blank'))
	assert await adapter.evaluate('1 + 1') == 2
	# Function-with-arg form (Playwright convention).
	assert await adapter.evaluate('(n) => n * 7', 6) == 42


async def test_playwright_adapter_has_no_accessibility_tree(playwright_page, httpserver) -> None:
	"""The Playwright backend must say it has no AX tree, not return an empty one.

	Playwright removed `page.accessibility` (deprecated, gone by 1.58 — the
	version the Firefox engine pins). The adapter used to walk frames inside a
	bare `except Exception: continue`, so the AttributeError was swallowed and
	callers got `{'nodes': []}` — indistinguishable from a page with no
	accessible nodes. Fail loudly instead; the agent loop reads interactive
	elements from the DOM + snapshot path, not from here.
	"""
	httpserver.expect_request('/page').respond_with_data(
		'<html><body><h1>Hello</h1><button>Click me</button><a href="#x">link</a></body></html>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/page'))

	for call in (adapter.accessibility_snapshot(), adapter.accessibility_snapshot_all_frames()):
		with pytest.raises(NotImplementedError, match='no accessibility tree'):
			await call

	# The page really does expose roles — proving the empty tree was the
	# adapter's bug, not an inaccessible fixture.
	outline = await playwright_page.locator('body').aria_snapshot()
	assert 'heading "Hello"' in outline
	assert 'button "Click me"' in outline


async def test_playwright_adapter_viewport_metrics(playwright_page, httpserver) -> None:
	"""Phase-3 ABC extension: device_pixel_ratio + viewport_metrics
	have natural cross-protocol meanings; both backends must implement."""
	httpserver.expect_request('/scroll').respond_with_data(
		'<html><body style="height: 4000px; width: 3000px;">scroll me</body></html>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/scroll'))

	dpr = await adapter.device_pixel_ratio()
	assert isinstance(dpr, float)
	assert dpr > 0

	vm = await adapter.viewport_metrics()
	# Contract: every key in the docstring is present and an int / float.
	for key in ('width', 'height', 'scroll_x', 'scroll_y', 'document_width', 'document_height', 'device_pixel_ratio'):
		assert key in vm, f'viewport_metrics missing {key!r}'
	assert vm['width'] > 0
	assert vm['height'] > 0
	# document_* must reflect the oversized body we served.
	assert vm['document_width'] >= 3000
	assert vm['document_height'] >= 4000
	assert vm['device_pixel_ratio'] == dpr
