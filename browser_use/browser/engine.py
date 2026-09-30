"""Browser engine abstraction — the seam between launch and protocol layers.

browser-use drives Chromium-family browsers over raw CDP (the cdp_use library).
Firefox does not speak CDP at all, so reaching Firefox — and with it Camoufox,
the anti-detect Firefox build — needs a second launch path behind a common
interface. This module is that interface.

  - :class:`BrowserEngine` — the contract: bring a browser up and expose
    whatever the protocol layer must connect to.
  - :class:`ChromiumCdpEngine` — the existing CDP path, delegating to
    :class:`~browser_use.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog`.
    No behaviour change from previous releases.
  - :class:`FirefoxPlaywrightEngine` — launches Camoufox (or stock Firefox)
    through the ``camoufox`` package and hands back a live Playwright object.
  - :func:`get_engine` — resolves a
    :class:`~browser_use.browser.profile.BrowserType` to an engine.

The two engines return different *kinds* of thing, and that asymmetry is
inherent rather than a rough edge. Chromium's ``launch()`` yields
``(process, cdp_url)`` because CDP is reached over a URL. A locally launched
Firefox has no such URL to hand out: Playwright Python exposes no
``launch_server()`` (that API is Node-only), so there is nothing to mint one
with. Firefox is therefore driven through the live Playwright object via
:meth:`FirefoxPlaywrightEngine.launch_local`, and
:meth:`FirefoxPlaywrightEngine.launch` raises with an explanation. A *remote*
Camoufox that already serves a Playwright ws endpoint is reached by handing
that endpoint to :class:`~browser_use.browser.connection.BidiBrowserConnection`
directly, bypassing this engine.

Quick local test:

    from browser_use.browser.engine import FirefoxPlaywrightEngine
    from browser_use.browser.profile import BrowserProfile

    profile = BrowserProfile(browser_type='firefox', headless=True)
    handle = await FirefoxPlaywrightEngine.launch_local(profile)
    try:
        page = await handle['browser'].new_page()
        await page.goto('https://books.toscrape.com')
        print((await page.content())[:200])
    finally:
        await handle['teardown']()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import psutil

from browser_use.browser.profile import BrowserProfile, BrowserType

if TYPE_CHECKING:
	from browser_use.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog


class BrowserEngine(ABC):
	"""The contract every browser engine implements.

	An *engine* owns the rules for bringing a browser subprocess up and
	exposing whatever the protocol layer needs to connect to it.

	:meth:`launch` is the URL-returning contract, and it fits Chromium only:
	CDP is reached over an endpoint, so ``(process, cdp_url)`` is the natural
	shape. Firefox has no equivalent — see
	:class:`FirefoxPlaywrightEngine`, whose :meth:`launch` raises and which
	offers :meth:`FirefoxPlaywrightEngine.launch_local` instead. Callers
	dispatch on ``profile.browser_type`` rather than treating the two as
	interchangeable.
	"""

	#: Stable short identifier — handy for log lines.
	name: str = 'base'

	@abstractmethod
	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		"""Launch a browser subprocess and return ``(process, cdp_url)``.

		``cdp_url`` is the endpoint the session layer connects to — for
		Chromium, the CDP HTTP+WS endpoint discovered via
		``http://localhost:<port>/json/version``.

		Raises :class:`NotImplementedError` on engines whose browser cannot
		be reached through a URL.
		"""
		raise NotImplementedError


class ChromiumCdpEngine(BrowserEngine):
	"""Chromium-family launch path — Chrome, Chromium, Edge, channels thereof.

	A thin pass-through to :meth:`LocalBrowserWatchdog._launch_browser`, which
	still owns the real subprocess + temp-dir + retry logic. The engine exists
	so both families are reachable through one interface; the Chromium launch
	itself has not moved.

	Nothing dispatches Chromium *through* here today — the watchdog calls its
	own ``_launch_browser`` directly. Keep it that way: routing the watchdog's
	launch back through this method would recurse.
	"""

	name = 'chromium'

	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		return await watchdog._launch_browser()


class FirefoxPlaywrightEngine(BrowserEngine):
	"""Firefox-family launch path — stock Firefox and Camoufox.

	Launches through ``camoufox.AsyncNewBrowser`` rather than driving
	Playwright directly. That is deliberate: the Camoufox binary is a
	Firefox build, so ``playwright.firefox.launch(executable_path=...)``
	*does* start it — but it starts it with none of Camoufox's
	configuration. The ``camoufox`` package is what builds ``CAMOU_CONFIG``
	from a fingerprint, attaches the addons, picks the virtual display and
	applies the ``no_viewport`` default that Camoufox's own window spoofing
	requires (daijro/camoufox#666). Skipping it yields a browser that is
	Camoufox in name only, and — because anti-bot clearance is bound to the
	identity that earned it — quietly invalidates the cookies in any profile
	a human has logged into.

	Two shapes come back from a launch, and the difference matters:

	  - ``user_data_dir`` set → a **persistent context**. Playwright returns a
	    ``BrowserContext`` with no ``Browser`` behind it, which is why
	    :class:`~browser_use.browser.connection.BidiBrowserConnection` has to
	    accept a context directly. This is the shape that reuses a logged-in
	    profile.
	  - no ``user_data_dir`` → a throwaway ``Browser``.

	There is no URL-returning launch here. The ``(process, url)`` contract on
	:class:`BrowserEngine` describes Chromium's CDP endpoint; a locally
	launched Firefox is reached through the live Playwright object instead.
	To attach to a *remote* Camoufox that already serves a Playwright ws
	endpoint, hand that endpoint to ``BidiBrowserConnection(ws_endpoint=...)``
	and skip this engine.
	"""

	name = 'firefox'

	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		"""Not available for Firefox: there is no connectable URL to return.

		Kept so the class still satisfies :class:`BrowserEngine`, and so a
		caller that reaches for the Chromium-shaped contract gets an
		explanation instead of a confusing failure further down.
		"""
		self.launch_url_contract_unavailable()
		raise AssertionError('unreachable')  # pragma: no cover

	def launch_url_contract_unavailable(self) -> None:
		"""Raise the explanation for why :meth:`launch` cannot work here."""
		raise NotImplementedError(
			'FirefoxPlaywrightEngine has no (process, url) launch: a locally launched '
			'Firefox/Camoufox is driven through the live Playwright object, not a URL. '
			'Use FirefoxPlaywrightEngine.launch_local(profile) and pass the result to '
			'BidiBrowserConnection, or give BidiBrowserConnection a ws_endpoint to reach '
			'a remote Camoufox. (Playwright Python has no launch_server(), so there is no '
			'way to mint a local endpoint either.)'
		)

	@staticmethod
	def build_launch_kwargs(profile: BrowserProfile) -> dict[str, Any]:
		"""Translate a FIREFOX profile into ``camoufox.AsyncNewBrowser`` kwargs.

		Pure, so the mapping can be asserted without starting a browser.
		``profile.camoufox_options`` is merged last: the caller owns the
		identity (fingerprint, config overlay, humanize, addons) and may
		correct anything derived here.
		"""
		if profile.browser_type != BrowserType.FIREFOX:
			raise ValueError(f'build_launch_kwargs expects a browser_type=firefox profile, got {profile.browser_type.value!r}')

		kwargs: dict[str, Any] = {'headless': bool(profile.headless)}

		if profile.executable_path:
			# Optional on purpose: left unset, the camoufox package resolves the
			# binary it fetched itself.
			kwargs['executable_path'] = str(profile.executable_path)

		if profile.user_data_dir:
			kwargs['persistent_context'] = True
			kwargs['user_data_dir'] = str(profile.user_data_dir)

		if profile.args:
			kwargs['args'] = list(profile.args)

		proxy = getattr(profile, 'proxy', None)
		if proxy is not None:
			proxy_dict = proxy.model_dump(exclude_none=True) if hasattr(proxy, 'model_dump') else dict(proxy)
			if proxy_dict:
				kwargs['proxy'] = proxy_dict

		kwargs.update(profile.camoufox_options or {})
		return kwargs

	@staticmethod
	async def launch_local(profile: BrowserProfile) -> dict[str, Any]:
		"""Launch Camoufox/Firefox locally and return a live Playwright handle.

		Returned shape::

		    {
		        'browser': Browser | None,  # None for a persistent context
		        'context': BrowserContext | None,  # None for a plain Browser
		        'playwright': Playwright,
		        'process': psutil.Process | None,
		        'teardown': Callable[[], Awaitable[None]],
		    }

		The caller owns the handle and must await ``teardown()``. Closing the
		Playwright object is what stops the browser; ``process`` is only there
		so a supervisor can force-kill a launch that will not close.
		"""
		from camoufox import AsyncNewBrowser
		from playwright.async_api import async_playwright

		kwargs = FirefoxPlaywrightEngine.build_launch_kwargs(profile)
		persistent = bool(kwargs.pop('persistent_context', False))

		pw = await async_playwright().start()
		try:
			launched = await AsyncNewBrowser(pw, persistent_context=persistent, **kwargs)
		except BaseException:
			# Never leak the Playwright runtime (and its node subprocess) when the
			# browser itself fails to come up.
			await pw.stop()
			raise

		browser = None if persistent else launched
		context = launched if persistent else None

		# Playwright exposes `.process` on a locally launched Browser. A persistent
		# context does not surface one, so a force-kill handle is best-effort here.
		raw = getattr(browser or context, 'process', None)
		process = psutil.Process(raw.pid) if raw is not None and getattr(raw, 'pid', None) else None

		async def _teardown() -> None:
			# Innermost first, and never raise: teardown runs on failure paths too.
			for step in (
				*((context.close,) if context is not None else ()),
				*((browser.close,) if browser is not None else ()),
				pw.stop,
			):
				try:
					await step()
				except Exception:  # noqa: BLE001 — best-effort teardown
					pass

		return {
			'browser': browser,
			'context': context,
			'playwright': pw,
			'process': process,
			'teardown': _teardown,
		}


# ── Factory ──────────────────────────────────────────────────────────────────


_ENGINES: dict[BrowserType, BrowserEngine] = {
	BrowserType.CHROMIUM: ChromiumCdpEngine(),
	BrowserType.FIREFOX: FirefoxPlaywrightEngine(),
}


def get_engine(browser_type: BrowserType) -> BrowserEngine:
	"""Return the engine matching ``browser_type``.

	Engines are stateless and shared across sessions — they hold no
	per-launch state, only the dispatch rules for that engine family.
	"""
	try:
		return _ENGINES[browser_type]
	except KeyError as e:
		raise ValueError(f'unknown browser_type: {browser_type!r}') from e
