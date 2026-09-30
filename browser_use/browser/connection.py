"""Browser connection — the bring-up + lifetime layer.

Where :mod:`adapter` abstracts *page-level operations* (goto, click, evaluate),
this module abstracts the *session-level resources* a
:class:`~browser_use.browser.session.BrowserSession` owns:

  - The long-lived control channel to the browser process (a CDP
    WebSocket for Chromium, a Playwright BiDi connection for Firefox).
  - Lifecycle (``start`` / ``stop`` of that channel).
  - Target discovery — listing tabs/pages, attaching to a specific one.

A :class:`BrowserConnection` is intentionally *narrower* than a full
Playwright Browser handle: it surfaces just enough for
:class:`BrowserSession` to bring itself up without leaking
protocol-specific types into the rest of browser-use.

Two implementations ship here:

  - :class:`CdpBrowserConnection` wraps the existing ``cdp_use.CDPClient``.
    Mirrors what :meth:`BrowserSession.connect` already does today —
    nothing functionally new on the Chromium path.

  - :class:`BidiBrowserConnection` wraps whatever Playwright handle reaches a
    Firefox / Camoufox browser — a locally launched persistent context, a
    locally launched Browser, or a remote ws endpoint. Same surface, so
    :class:`~browser_use.browser.session.BrowserSession` brings up the same way
    on either engine.

Watchdogs have not been refactored to consume the connection; they still call
raw ``session.cdp_client.send.*``. On the BiDi backend those calls land on
:class:`~browser_use.browser.cdp_proxy.BidiCdpProxy`, which translates them into
Playwright. See ``ADAPTERS.md`` for the boundary doctrine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
	# Type-checking only, so these imports never run — playwright stays an
	# optional dependency for CDP-only deploys (a try/except fallback here would
	# make each name a variable, which is not usable in an annotation).
	from cdp_use import CDPClient
	from playwright.async_api import Browser as PlaywrightBrowser
	from playwright.async_api import BrowserContext as PlaywrightContext
	from playwright.async_api import Page as PlaywrightPage
	from playwright.async_api import Playwright


class BrowserConnection(ABC):
	"""The contract every browser-control connection implements.

	An instance owns the network channel + lifecycle of one logical
	browser. Long-lived: created on session start, torn down on session
	stop.

	The :attr:`backend` attribute returns a stable short identifier
	(``'cdp'`` or ``'bidi'``) so caller code can branch when it has to
	(legacy paths that haven't been refactored to the
	:class:`~browser_use.browser.adapter.BrowserAdapter` yet).
	"""

	#: Stable identifier — ``'cdp'`` or ``'bidi'``.
	backend: str = 'base'

	@abstractmethod
	async def start(self) -> None:
		"""Open the underlying channel. Idempotent: safe to call twice."""

	@abstractmethod
	async def stop(self) -> None:
		"""Tear down the channel. Idempotent: safe to call after start
		failed or after a previous stop."""

	@property
	@abstractmethod
	def is_open(self) -> bool:
		"""``True`` between a successful :meth:`start` and the first
		:meth:`stop` (or an underlying connection failure)."""


# ── CDP backend ──────────────────────────────────────────────────────────────


class CdpBrowserConnection(BrowserConnection):
	"""Wraps :class:`cdp_use.CDPClient`. Used by every Chromium target.

	The connection is constructed *around* an existing :class:`CDPClient`
	instance — that matches how :class:`BrowserSession.connect` currently
	builds the client (with retry, timeout, additional_headers), and lets
	this class stay agnostic about *how* the client was built.
	"""

	backend = 'cdp'

	def __init__(self, cdp_client: CDPClient) -> None:
		self._client = cdp_client
		self._started = False

	@property
	def client(self) -> CDPClient:
		"""Access the underlying :class:`CDPClient` — used by legacy
		paths in :mod:`browser_use.browser.session` and the watchdogs
		that haven't been refactored to the
		:class:`~browser_use.browser.adapter.BrowserAdapter` yet."""
		return self._client

	async def start(self) -> None:
		if self._started:
			return
		await self._client.start()
		self._started = True

	async def stop(self) -> None:
		if not self._started:
			return
		try:
			await self._client.stop()
		finally:
			self._started = False

	@property
	def is_open(self) -> bool:
		if not self._started:
			return False
		ws = getattr(self._client, 'ws', None)
		if ws is None:
			return False
		# cdp_use uses an internal State enum on its WS wrapper. Coerce
		# defensively so a refactor of cdp_use's internals doesn't break
		# this method.
		state = getattr(ws, 'state', None)
		return bool(state) and str(state).upper().endswith('OPEN')


# ── BiDi (Playwright) backend ────────────────────────────────────────────────


class BidiBrowserConnection(BrowserConnection):
	"""Wraps a Playwright handle on a Firefox / Camoufox browser.

	Construction takes exactly one of three entry points, and the class hides
	which was used from everything downstream:

	  - ``context`` — a persistent :class:`BrowserContext` from
	    :meth:`~browser_use.browser.engine.FirefoxPlaywrightEngine.launch_local`
	    with a ``user_data_dir``. Playwright backs a persistent context with no
	    ``Browser``, so the context is the whole connection. This is the shape
	    that reuses an already-logged-in profile.
	  - ``browser`` — a locally launched, throwaway :class:`Browser`.
	  - ``ws_endpoint`` — a remote Camoufox already serving Playwright.

	Two-step lifecycle:

	1. ``__init__`` records inputs but does NOT touch the network.
	2. :meth:`start` enters async-land: starts the Playwright runtime if we own
	   one, connects to ``ws_endpoint`` if given, settles on a
	   :class:`BrowserContext` and picks one :class:`Page`. That page becomes the
	   "current tab" watchdogs reach through :attr:`current_page` — a single-tab
	   posture until multi-tab tracking lands.
	"""

	backend = 'bidi'

	def __init__(
		self,
		*,
		browser: PlaywrightBrowser | None = None,
		context: PlaywrightContext | None = None,
		playwright: Playwright | None = None,
		ws_endpoint: str | None = None,
		proxy: dict | None = None,
		process: Any | None = None,
	) -> None:
		if browser is None and context is None and ws_endpoint is None:
			raise ValueError(
				'BidiBrowserConnection requires one of `browser` (a connected Playwright '
				'Browser), `context` (a persistent BrowserContext) or `ws_endpoint` '
				'(a remote Camoufox to connect to)'
			)
		self._browser: PlaywrightBrowser | None = browser
		self._playwright: Playwright | None = playwright
		self._ws_endpoint = ws_endpoint
		# A persistent-context launch (user_data_dir — the shape that reuses a
		# logged-in profile) returns a BrowserContext with no Browser behind it,
		# so the context *is* the whole connection: start() adopts it instead of
		# calling new_context(), and stop() must not reach for a Browser.
		self._injected_context = context
		# Best-effort force-kill handle for a browser that refuses to close.
		# Playwright does not surface a process for a persistent context, so this
		# is often None even on a local launch.
		self.process = process
		# Per-CONTEXT proxy (geo / egress IP), applied CLIENT-SIDE: even when we
		# connect to a remote Camoufox over the ws, Playwright Firefox honours a
		# proxy passed to new_context(). dict form: {server, username, password, bypass}.
		if proxy and context is not None:
			# A per-context proxy has to be set when the context is created, and a
			# persistent context already exists by the time we see it. Refuse rather
			# than accept a proxy we would silently drop — the caller would think
			# their egress IP had changed.
			raise ValueError(
				'BidiBrowserConnection cannot apply `proxy` to an injected `context`: a '
				'proxy is per-context and must be set at creation. Pass the proxy to the '
				'launch instead (BrowserProfile.proxy reaches camoufox), or use the '
				'`browser`/`ws_endpoint` form so a fresh context can carry it.'
			)
		self._proxy = proxy
		self._owns_playwright = playwright is None and browser is None and context is None
		self._started = False
		# Default context + page — populated by start(). Single-tab posture
		# for the first Firefox iteration; multi-tab discovery becomes
		# the BiDi analog of SessionManager when needed.
		self._context: PlaywrightContext | None = context
		self._page: PlaywrightPage | None = None

	@property
	def browser(self) -> PlaywrightBrowser:
		"""Access the underlying Playwright Browser. Raises if there isn't one.

		A persistent-context connection has no Browser — Playwright does not
		expose one for ``launch_persistent_context``. Use :attr:`context`.
		"""
		if self._browser is None:
			if self._injected_context is not None:
				raise RuntimeError(
					'BidiBrowserConnection has no Browser: it was built from a persistent '
					'BrowserContext, which Playwright does not back with one. Use .context'
				)
			raise RuntimeError('BidiBrowserConnection: start() not called yet')
		return self._browser

	@property
	def playwright(self) -> Playwright | None:
		"""The Playwright runtime — ``None`` when the caller injected an
		already-connected Browser and kept the runtime under its own
		control."""
		return self._playwright

	async def start(self) -> None:
		if self._started and self._page is not None:
			return
		from playwright.async_api import async_playwright

		# An injected persistent context is already the connection — no runtime to
		# start, nothing to connect to, and no context to create. Just pick the tab.
		if self._injected_context is None:
			if self._playwright is None:
				self._playwright = await async_playwright().start()

			browser = self._browser
			if browser is None:
				assert self._ws_endpoint is not None, 'ws_endpoint required when no Browser injected'
				browser = await self._playwright.firefox.connect(self._ws_endpoint)
				self._browser = browser

			# Context selection. With a proxy we MUST create a fresh context that
			# carries it (geo/egress IP is per-context) — the default context that
			# Camoufox already opened can't be re-proxied after the fact. Without a
			# proxy, reuse the existing context (cheaper) or make a default one.
			if self._proxy:
				self._context = await browser.new_context(proxy=cast('Any', self._proxy))
			elif browser.contexts:
				self._context = browser.contexts[0]
			else:
				self._context = await browser.new_context()

		assert self._context is not None
		if self._context.pages:
			self._page = self._context.pages[0]
		else:
			self._page = await self._context.new_page()

		self._started = True

	async def stop(self) -> None:
		# Order: page → context → browser → playwright runtime. Each step
		# is best-effort and never raises. We close the context only when
		# we created it (i.e. when starting from a fresh Browser without
		# pre-existing contexts).
		if self._page is not None:
			try:
				await self._page.close()
			except Exception:
				pass
			self._page = None
		if self._context is not None:
			try:
				await self._context.close()
			except Exception:
				pass
			self._context = None
		if self._browser is not None:
			try:
				await self._browser.close()
			except Exception:
				pass
			self._browser = None
		if self._owns_playwright and self._playwright is not None:
			try:
				await self._playwright.stop()
			except Exception:
				pass
			self._playwright = None
		self._started = False

	@property
	def is_open(self) -> bool:
		if not self._started:
			return False
		if self._browser is not None:
			# Playwright Browser exposes `is_connected()` for liveness.
			return bool(self._browser.is_connected())
		if self._context is not None:
			# A persistent context has no is_connected(). Its `browser` is None by
			# construction, so liveness is "the context still answers" — stop()
			# clears _context, and a context whose browser died raises on use.
			try:
				return self._page is not None and not self._page.is_closed()
			except Exception:  # noqa: BLE001 — a dead context/page means not open
				return False
		return False

	# ── Page / context accessors (used by watchdogs going through the
	# Playwright path; CDP-shaped watchdogs continue using cdp_client). ─

	@property
	def context(self) -> PlaywrightContext:
		"""Default :class:`BrowserContext`. Raises if connection isn't started."""
		if self._context is None:
			raise RuntimeError('BidiBrowserConnection: start() not called yet')
		return self._context

	@property
	def current_page(self) -> PlaywrightPage:
		"""The "active tab" :class:`Page` watchdogs interact with. Until
		multi-tab tracking lands this is the single default page opened
		by :meth:`start`. Watchdogs that need a different page MUST
		switch the context's active page first (Playwright's
		``page.bring_to_front``) then re-fetch."""
		if self._page is None:
			raise RuntimeError('BidiBrowserConnection: no active page (start() not called?)')
		return self._page


# ── Factory helpers ──────────────────────────────────────────────────────────


def connection_from_browser_type(browser_type: str) -> type[BrowserConnection]:
	"""Resolve a :class:`~browser_use.browser.profile.BrowserType`-shaped
	string (``'chromium'`` / ``'firefox'``) to the connection class that
	wraps that backend's control protocol."""
	if browser_type == 'chromium':
		return CdpBrowserConnection
	if browser_type == 'firefox':
		return BidiBrowserConnection
	raise ValueError(f'no BrowserConnection mapped for browser_type={browser_type!r}')


__all__ = [
	'BrowserConnection',
	'CdpBrowserConnection',
	'BidiBrowserConnection',
	'connection_from_browser_type',
]
