"""Firefox/Camoufox engine: launch-kwargs construction and the local launch.

The ported engine originally called ``playwright.firefox.launch_server()``,
which is a Node-only Playwright API — it does not exist in Playwright Python
(``BrowserType`` exposes only connect, connect_over_cdp, launch and
launch_persistent_context). The upstream fork validated its Firefox path by
connecting to a remote Camoufox pod over a ws endpoint, so the local-launch
branch was never exercised.

It also launched the Camoufox *binary* through plain Playwright, bypassing the
``camoufox`` package entirely: no CAMOU_CONFIG, no fingerprint, no humanize, no
addons, no virtual display. That matters beyond stealth — clearance cookies are
bound to the fingerprint that earned them, so launching the binary raw
invalidates a profile a human has logged in.

Both are covered here: the kwargs builder is a pure function so it can be
asserted without a browser, and one opt-in test does a real launch.
"""

import ast
import inspect
from pathlib import Path

import pytest

from browser_use.browser.engine import (
	BrowserEngine,
	ChromiumCdpEngine,
	FirefoxPlaywrightEngine,
	get_engine,
)
from browser_use.browser.profile import BrowserProfile, BrowserType

# ── Regression guards on the Node-only API ───────────────────────────────────


def test_engine_never_calls_the_node_only_launch_server():
	"""`launch_server` is Node-only; reaching for it again would break at runtime.

	Parsed rather than grepped so the docstrings that *explain* the absence
	don't trip the guard.
	"""
	source = Path(inspect.getfile(FirefoxPlaywrightEngine)).read_text(encoding='utf-8')
	tree = ast.parse(source)
	offending = [
		f'line {node.lineno}' for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == 'launch_server'
	]
	assert not offending, f'launch_server is absent from Playwright Python: {offending}'


def test_playwright_python_really_lacks_launch_server():
	"""Pin the assumption behind the guard above, so a Playwright bump surfaces it."""
	playwright = pytest.importorskip('playwright.async_api')
	assert not hasattr(playwright.BrowserType, 'launch_server')
	for expected in ('launch', 'launch_persistent_context', 'connect', 'connect_over_cdp'):
		assert hasattr(playwright.BrowserType, expected)


def test_abc_launch_is_unavailable_for_firefox_and_says_why():
	"""The (process, url) contract can't describe a locally launched Camoufox."""
	with pytest.raises(NotImplementedError, match='launch_local'):
		FirefoxPlaywrightEngine().launch_url_contract_unavailable()


# ── The kwargs builder ───────────────────────────────────────────────────────


def test_build_launch_kwargs_maps_profile_fields():
	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=True,
		executable_path='/opt/camoufox/camoufox',
		# BrowserLaunchArgs.args validates Chromium style, so `--`-prefixed even
		# though Firefox's own flags are conventionally single-dash.
		args=['--marionette'],
	)
	kwargs = FirefoxPlaywrightEngine.build_launch_kwargs(profile)
	assert kwargs['headless'] is True
	assert kwargs['executable_path'] == '/opt/camoufox/camoufox'
	assert kwargs['args'] == ['--marionette']
	# No profile dir means a throwaway browser, not a persistent context.
	assert 'user_data_dir' not in kwargs
	assert kwargs.get('persistent_context', False) is False


def test_build_launch_kwargs_requests_a_persistent_context_for_a_profile_dir(tmp_path):
	profile = BrowserProfile(browser_type=BrowserType.FIREFOX, headless=True, user_data_dir=str(tmp_path / 'ud'))
	kwargs = FirefoxPlaywrightEngine.build_launch_kwargs(profile)
	assert kwargs['persistent_context'] is True
	assert kwargs['user_data_dir'] == str(tmp_path / 'ud')


def test_build_launch_kwargs_omits_an_unset_executable_path():
	"""Left unset, the camoufox package resolves its own fetched binary."""
	profile = BrowserProfile(browser_type=BrowserType.FIREFOX, headless=True)
	assert 'executable_path' not in FirefoxPlaywrightEngine.build_launch_kwargs(profile)


def test_build_launch_kwargs_passes_the_proxy_through():
	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=True,
		proxy={'server': 'http://p:8080', 'username': 'u', 'password': 'pw'},
	)
	kwargs = FirefoxPlaywrightEngine.build_launch_kwargs(profile)
	assert kwargs['proxy'] == {'server': 'http://p:8080', 'username': 'u', 'password': 'pw'}


def test_camoufox_options_override_the_derived_kwargs():
	"""The pass-through seam: identity config the engine must not have to know about.

	Callers own the fingerprint, config overlay, humanize and addons (for us,
	edwin's pinned identity). Later keys win so a caller can correct anything
	derived from the profile.
	"""
	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=True,
		camoufox_options={'humanize': True, 'i_know_what_im_doing': True, 'headless': False},
	)
	kwargs = FirefoxPlaywrightEngine.build_launch_kwargs(profile)
	assert kwargs['humanize'] is True
	assert kwargs['i_know_what_im_doing'] is True
	assert kwargs['headless'] is False, 'camoufox_options must win over the profile'


def test_build_launch_kwargs_rejects_a_chromium_profile():
	profile = BrowserProfile(browser_type=BrowserType.CHROMIUM, headless=True)
	with pytest.raises(ValueError, match='firefox'):
		FirefoxPlaywrightEngine.build_launch_kwargs(profile)


# ── Factory ──────────────────────────────────────────────────────────────────


def test_factory_returns_the_matching_engine():
	assert isinstance(get_engine(BrowserType.CHROMIUM), ChromiumCdpEngine)
	assert isinstance(get_engine(BrowserType.FIREFOX), FirefoxPlaywrightEngine)
	assert isinstance(get_engine(BrowserType.FIREFOX), BrowserEngine)


# ── A real launch ────────────────────────────────────────────────────────────


async def test_launch_local_brings_up_a_live_camoufox_page(tmp_path):
	"""End-to-end: the engine launches Camoufox and hands back a usable page.

	Skips when camoufox isn't installed or its Firefox hasn't been fetched.
	"""
	pytest.importorskip('camoufox')
	pytest.importorskip('playwright.async_api')

	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=True,
		user_data_dir=str(tmp_path / 'camoufox-profile'),
	)
	try:
		handle = await FirefoxPlaywrightEngine.launch_local(profile)
	except Exception as e:  # noqa: BLE001 — a missing binary must skip, not fail
		pytest.skip(f'camoufox could not launch: {e}')

	try:
		# A persistent context launch yields a BrowserContext, never a Browser.
		assert handle['context'] is not None
		assert handle['browser'] is None, 'persistent_context has no Browser handle'
		page = handle['context'].pages[0] if handle['context'].pages else await handle['context'].new_page()
		await page.goto('data:text/html,<title>engine</title><h1>up</h1>')
		assert await page.title() == 'engine'
		assert handle['playwright'] is not None
	finally:
		await handle['teardown']()


async def test_bidi_connection_accepts_an_injected_context(tmp_path):
	"""A persistent context has no Browser, so the connection must take it directly.

	This is what lets a Camoufox run reuse an already-logged-in profile: the
	upstream fork only knew how to `firefox.connect(ws_endpoint)`.
	"""
	pytest.importorskip('camoufox')
	from browser_use.browser.connection import BidiBrowserConnection

	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=True,
		user_data_dir=str(tmp_path / 'camoufox-profile'),
	)
	try:
		handle = await FirefoxPlaywrightEngine.launch_local(profile)
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'camoufox could not launch: {e}')

	conn = BidiBrowserConnection(context=handle['context'], playwright=handle['playwright'], process=handle['process'])
	try:
		await conn.start()
		assert conn.backend == 'bidi'
		assert conn.is_open
		await conn.current_page.goto('data:text/html,<title>injected</title>')
		assert await conn.current_page.title() == 'injected'
	finally:
		await conn.stop()
		assert not conn.is_open


def test_bidi_connection_needs_one_of_browser_context_or_ws_endpoint():
	from browser_use.browser.connection import BidiBrowserConnection

	with pytest.raises(ValueError, match='ws_endpoint'):
		BidiBrowserConnection()
