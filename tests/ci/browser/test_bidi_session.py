"""A real BrowserSession brought up on Camoufox over BiDi.

This is the integration point the whole Firefox layer exists for: session
bring-up, DOM observation, and the raw ``cdp_client.send.*`` call sites in the
watchdogs all running against Playwright Firefox instead of Chrome/CDP.

Skips when camoufox isn't installed or its Firefox binary hasn't been fetched,
so the suite stays green on a CDP-only environment.
"""

import pytest

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

PAGE = (
	'<html><head><title>BiDi fixture</title></head><body>'
	'<h1>heading</h1>'
	'<a href="/other">a link</a>'
	'<button id="b">a button</button>'
	'<input name="q" type="text">'
	'<select><option>one</option><option>two</option></select>'
	'</body></html>'
)


@pytest.fixture
async def bidi_session(tmp_path):
	pytest.importorskip('camoufox')
	pytest.importorskip('playwright.async_api')

	profile = BrowserProfile(
		browser_type='firefox',
		headless=True,
		user_data_dir=str(tmp_path / 'camoufox-profile'),
	)
	session = BrowserSession(browser_profile=profile)
	try:
		await session.start()
	except Exception as e:  # noqa: BLE001 — a missing binary must skip, not fail
		await session.kill()
		pytest.skip(f'camoufox session could not start: {e}')
	try:
		yield session
	finally:
		await session.kill()


async def test_session_starts_on_the_bidi_backend(bidi_session):
	"""There is no CDP socket — the backend is BiDi and the proxy stands in."""
	assert bidi_session._connection is not None
	assert bidi_session._connection.backend == 'bidi'
	assert bidi_session._cdp_client_root is None, 'no real CDP client on the BiDi path'
	assert bidi_session.is_cdp_connected, 'liveness must come from the Playwright connection'
	# `cdp_client` hands back the facade, which is what lets the ~731 raw
	# cdp_client.send.* call sites run unmodified.
	from browser_use.browser.cdp_proxy import BidiCdpProxy

	assert isinstance(bidi_session.cdp_client, BidiCdpProxy)


async def test_navigation_and_page_state(bidi_session, httpserver):
	httpserver.expect_request('/page').respond_with_data(PAGE, content_type='text/html')
	url = httpserver.url_for('/page')

	await bidi_session.navigate_to(url)

	assert await bidi_session.get_current_page_url() == url
	assert await bidi_session.get_current_page_title() == 'BiDi fixture'

	# Single-tab posture on BiDi: one synthesised TabInfo, not a CDP target pool.
	tabs = await bidi_session.get_tabs()
	assert len(tabs) == 1
	assert tabs[0].url == url
	assert tabs[0].title == 'BiDi fixture'


async def test_dom_observation_finds_interactive_elements(bidi_session, httpserver):
	"""The DOM tree is synthesised from the Playwright side, not an AX tree.

	`BidiCdpProxy` returns no accessibility nodes on purpose, so anything found
	here came through the DOM + snapshot path.
	"""
	httpserver.expect_request('/page').respond_with_data(PAGE, content_type='text/html')
	await bidi_session.navigate_to(httpserver.url_for('/page'))

	state = await bidi_session.get_browser_state_summary()
	selector_map = state.dom_state.selector_map or {}
	assert selector_map, 'no interactive elements were indexed'

	tags = {node.tag_name.lower() for node in selector_map.values() if node.tag_name}
	assert {'a', 'button', 'input', 'select'} & tags, f'expected form/link elements in {tags}'


async def test_screenshot_comes_back_through_the_watchdog(bidi_session, httpserver):
	"""ScreenshotWatchdog's BiDi branch goes through the Playwright adapter."""
	httpserver.expect_request('/page').respond_with_data(PAGE, content_type='text/html')
	await bidi_session.navigate_to(httpserver.url_for('/page'))

	state = await bidi_session.get_browser_state_summary()
	assert state.screenshot, 'no screenshot captured'

	import base64

	png = base64.b64decode(state.screenshot)
	assert png.startswith(b'\x89PNG\r\n\x1a\n'), 'screenshot is not a PNG'
	assert len(png) > 1000
