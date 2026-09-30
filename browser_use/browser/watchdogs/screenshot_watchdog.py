"""Screenshot watchdog — dual-mode CDP / Playwright via BrowserAdapter."""

import asyncio
import base64
import os
from typing import TYPE_CHECKING, Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.page import CaptureScreenshotParameters

from browser_use.browser.events import ScreenshotEvent
from browser_use.browser.views import BrowserError
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.observability import observe_debug

if TYPE_CHECKING:
	pass


class ScreenshotWatchdog(BaseWatchdog):
	"""Handles screenshot requests on both CDP and BiDi backends.

	Phase-5b port. The Chromium path (CDP) is bit-identical to the
	previous implementation. On BiDi (Firefox/Camoufox) we go through
	the :class:`PlaywrightBrowserAdapter` which has the
	:meth:`screenshot` method on the page-level contract.

	Caveat for the BiDi path: ``event.clip`` is honoured only on the
	CDP backend today — Playwright's ``page.screenshot(clip=…)`` exists
	but uses a slightly different coordinate origin and we haven't
	verified parity yet. When ``event.clip`` is set on BiDi we still
	take a full-page / full-viewport screenshot and log a warning.
	"""

	DEFAULT_CAPTURE_COMMAND_TIMEOUT_SECONDS: ClassVar[float] = 10.0
	CAPTURE_COMMAND_TIMEOUT_ENV: ClassVar[str] = 'BROWSER_USE_SCREENSHOT_COMMAND_TIMEOUT_SECONDS'
	MAX_CAPTURE_RECOVERY_RETRIES: ClassVar[int] = 1

	# Events this watchdog listens to
	LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [ScreenshotEvent]

	# Events this watchdog emits
	EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

	def _resolve_capture_command_timeout_seconds(self, event: ScreenshotEvent) -> float:
		configured_timeout = self.DEFAULT_CAPTURE_COMMAND_TIMEOUT_SECONDS
		env_value = os.environ.get(self.CAPTURE_COMMAND_TIMEOUT_ENV)
		if env_value:
			try:
				parsed = float(env_value)
				if parsed > 0:
					configured_timeout = parsed
			except ValueError:
				self.logger.debug(f'[ScreenshotWatchdog] Ignoring invalid {self.CAPTURE_COMMAND_TIMEOUT_ENV}={env_value!r}')

		event_timeout = float(event.event_timeout) if event.event_timeout is not None else None
		if event_timeout is None:
			return configured_timeout

		# Leave headroom for reconnect/retry and for the event bus bookkeeping timeout.
		return max(0.01, min(configured_timeout, max(event_timeout - 2.0, 0.01)))

	def _is_retryable_capture_error(self, exc: Exception) -> bool:
		if isinstance(exc, TimeoutError):
			return True

		error_text = str(exc).lower()
		retryable_markers = (
			'client is not started',
			'connection closed',
			'not connected',
			'session with given id not found',
			'target closed',
		)
		return any(marker in error_text for marker in retryable_markers)

	def _resolve_screenshot_target_id(self) -> str:
		focused_target = self.browser_session.get_focused_target()

		if focused_target and focused_target.target_type in ('page', 'tab'):
			return focused_target.target_id

		target_type_str = focused_target.target_type if focused_target else 'None'
		self.logger.warning(f'[ScreenshotWatchdog] Focused target is {target_type_str}, falling back to page target')
		page_targets = self.browser_session.get_page_targets()
		if not page_targets:
			raise BrowserError('[ScreenshotWatchdog] No page targets available for screenshot')
		return page_targets[-1].target_id

	@staticmethod
	def _build_capture_params(event: ScreenshotEvent) -> CaptureScreenshotParameters:
		params_dict: dict[str, Any] = {'format': 'png', 'captureBeyondViewport': event.full_page}
		if event.clip:
			params_dict['clip'] = {
				'x': event.clip['x'],
				'y': event.clip['y'],
				'width': event.clip['width'],
				'height': event.clip['height'],
				'scale': 1,
			}
		return CaptureScreenshotParameters(**params_dict)

	async def _capture_screenshot_once(
		self,
		*,
		target_id: str,
		params: CaptureScreenshotParameters,
		timeout_seconds: float,
	) -> str:
		cdp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=True)
		result = await asyncio.wait_for(
			cdp_session.cdp_client.send.Page.captureScreenshot(
				params=params,
				session_id=cdp_session.session_id,
			),
			timeout=timeout_seconds,
		)

		if result and 'data' in result:
			self.logger.debug('[ScreenshotWatchdog] Screenshot captured successfully')
			return result['data']

		raise BrowserError('[ScreenshotWatchdog] Screenshot result missing data')

	async def _recover_after_capture_failure(self, target_id: str, exc: Exception) -> None:
		self.logger.warning(
			f'[ScreenshotWatchdog] Screenshot capture failed for target {target_id[:8]}... '
			f'with {type(exc).__name__}: {exc}. Reconnecting CDP and retrying once.'
		)
		await self.browser_session.reconnect()
		await self.browser_session.get_or_create_cdp_session(target_id, focus=True)

	@observe_debug(ignore_input=True, ignore_output=True, name='screenshot_event_handler')
	async def on_ScreenshotEvent(self, event: ScreenshotEvent) -> str:
		"""Handle screenshot request. Dispatches by backend.

		Returns:
			Base64-encoded PNG screenshot data (no `data:` URL prefix).
		"""
		# Phase-5b dispatch: route to the BiDi handler when the session
		# is on a Playwright/Firefox backend. Falls through to the
		# legacy CDP path otherwise.
		conn = getattr(self.browser_session, '_connection', None)
		if conn is not None and conn.backend == 'bidi':
			return await self._on_screenshot_bidi(event)
		return await self._on_screenshot_cdp(event)

	async def _on_screenshot_cdp(self, event: ScreenshotEvent) -> str:
		"""Legacy CDP path — bit-identical to the pre-Phase-5b implementation."""
		self.logger.debug('[ScreenshotWatchdog] (CDP) Handler START - on_ScreenshotEvent called')
		try:
			# Remove highlights BEFORE taking the screenshot so they don't appear in the image.
			# Done here (not in finally) so CancelledError is never swallowed — any await in a
			# finally block can suppress external task cancellation.
			# remove_highlights() has its own asyncio.timeout(3.0) internally so it won't block.
			try:
				await self.browser_session.remove_highlights()
			except Exception:
				pass

			target_id = self._resolve_screenshot_target_id()
			params = self._build_capture_params(event)
			timeout_seconds = self._resolve_capture_command_timeout_seconds(event)

			self.logger.debug(f'[ScreenshotWatchdog] Taking screenshot with params: {params}')
			last_error: Exception | None = None
			for attempt in range(self.MAX_CAPTURE_RECOVERY_RETRIES + 1):
				try:
					return await self._capture_screenshot_once(
						target_id=target_id,
						params=params,
						timeout_seconds=timeout_seconds,
					)
				except Exception as exc:
					last_error = exc
					if attempt >= self.MAX_CAPTURE_RECOVERY_RETRIES or not self._is_retryable_capture_error(exc):
						break
					await self._recover_after_capture_failure(target_id, exc)

			if isinstance(last_error, TimeoutError):
				raise BrowserError(
					f'[ScreenshotWatchdog] Screenshot capture timed out after {timeout_seconds:.1f}s'
				) from last_error
			if last_error is not None:
				raise last_error
			raise BrowserError('[ScreenshotWatchdog] Screenshot failed without a captured error')
		except Exception as e:
			self.logger.error(f'[ScreenshotWatchdog] (CDP) Screenshot failed: {e}')
			raise

	async def _on_screenshot_bidi(self, event: ScreenshotEvent) -> str:
		"""BiDi (Playwright Firefox/Camoufox) path via BrowserAdapter."""
		self.logger.debug('[ScreenshotWatchdog] (BiDi) Handler START - on_ScreenshotEvent called')
		try:
			# Try removing highlights first — same posture as the CDP path.
			# The method is CDP-bound on the current codebase and no-ops
			# (or raises silently) on BiDi; suppress and continue.
			try:
				await self.browser_session.remove_highlights()
			except Exception:
				pass

			if event.clip:
				self.logger.warning(
					'[ScreenshotWatchdog] (BiDi) `event.clip` not yet honoured on the '
					'Playwright path — taking a full-viewport screenshot instead'
				)

			adapter = await self.browser_session.get_adapter()
			png_bytes = await adapter.screenshot(full_page=bool(event.full_page), fmt='png')
			# Return base64-encoded PNG (no data: prefix) to match the CDP path's contract.
			b64 = base64.b64encode(png_bytes).decode('ascii')
			self.logger.debug('[ScreenshotWatchdog] (BiDi) Screenshot captured successfully')
			return b64
		except Exception as e:
			self.logger.error(f'[ScreenshotWatchdog] (BiDi) Screenshot failed: {e}')
			raise
