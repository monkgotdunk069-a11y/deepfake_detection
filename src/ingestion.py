"""Safe-ish ingestion helpers for user supplied public video URLs.

Direct media URLs are fetched with the pinned HTTP client. Public platform
pages may be resolved with yt-dlp; private, login-protected, DRM, and
authentication-required pages are rejected.
"""

import ipaddress
import http.client
import os
import socket
import ssl
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse


MAX_VIDEO_BYTES = 500 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mpeg", ".mpg", ".m4v"}
VIDEO_CONTENT_TYPES = {"application/octet-stream", "binary/octet-stream"}


def _validate_public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Use a complete public http:// or https:// video URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not accepted.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The URL has an invalid port.") from exc
    if port not in {None, 80, 443}:
        raise ValueError("Only standard public web ports (80 and 443) are accepted.")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("The video host could not be resolved.") from exc
    if not addresses:
        raise ValueError("The video host did not resolve to an address.")
    for address in {item[4][0] for item in addresses}:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global:
            raise ValueError("Private, local, and reserved network addresses are not allowed.")
    return parsed, sorted({item[4][0].split("%", 1)[0] for item in addresses})


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, ip, **kwargs):
        super().__init__(host, port, **kwargs)
        self._pinned_ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout, self.source_address)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, port, ip, **kwargs):
        super().__init__(host, port, **kwargs)
        self._pinned_ip = ip

    def connect(self):
        raw_socket = socket.create_connection((self._pinned_ip, self.port), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


def _open_public_response(url, parsed, public_ips):
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_type = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    last_error = None
    for ip in public_ips:
        connection = connection_type(parsed.hostname, port, ip, timeout=20, context=ssl.create_default_context()) if parsed.scheme == "https" else connection_type(parsed.hostname, port, ip, timeout=20)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            connection.request("GET", path, headers={
                "Host": parsed.netloc,
                "User-Agent": "DeepDetectAI/1.0",
                "Accept": "video/*,application/octet-stream",
                "Connection": "close",
            })
            response = connection.getresponse()
            response._deepdetect_connection = connection
            return response
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            last_error = exc
    raise ValueError(f"Could not connect to the public video host: {last_error}")


def download_video_url(url, max_bytes=MAX_VIDEO_BYTES):
    """Download a direct public video URL to a temporary file.

    Redirects are handled manually so each destination is checked before use.
    Returns ``(path, filename, final_url)``. The caller owns cleanup of path.
    """
    current_url = url.strip()
    response = None
    for redirect_count in range(6):
        parsed, public_ips = _validate_public_url(current_url)
        response = _open_public_response(current_url, parsed, public_ips)
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            response.close()
            response._deepdetect_connection.close()
            response = None
            if not location or redirect_count >= 5:
                raise ValueError("The video URL redirected too many times or had an invalid redirect.")
            current_url = urljoin(current_url, location)
            continue
        if response.status >= 400:
            status = response.status
            response.close()
            response._deepdetect_connection.close()
            response = None
            raise ValueError(f"The video host returned HTTP {status}.")
        break
    if response is None:
        raise ValueError("Could not open the video URL.")

    try:
        content_type = response.getheader("Content-Type", "application/octet-stream").split(";", 1)[0].strip().lower()
        if not content_type.startswith("video/") and content_type not in VIDEO_CONTENT_TYPES:
            response.close()
            response._deepdetect_connection.close()
            return _download_public_platform_video(url, max_bytes)
        content_length = response.getheader("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"The video exceeds the {max_bytes // (1024 * 1024)} MB import limit.")

        filename = os.path.basename(urlparse(current_url).path) or "linked-video"
        if "." not in filename:
            filename += ".mp4" if content_type == "video/mp4" else ".video"
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in VIDEO_EXTENSIONS:
            if content_type == "video/mp4":
                suffix = ".mp4"
            elif content_type == "video/quicktime":
                suffix = ".mov"
            elif content_type == "video/webm":
                suffix = ".webm"
            elif content_type == "video/x-msvideo":
                suffix = ".avi"
            else:
                suffix = ".mp4"
            filename = os.path.splitext(filename)[0] + suffix

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="deepdetect_url_")
        total = 0
        try:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"The video exceeds the {max_bytes // (1024 * 1024)} MB import limit.")
                temp.write(chunk)
            if total == 0:
                raise ValueError("The video URL returned an empty file.")
        except Exception:
            temp.close()
            try:
                os.unlink(temp.name)
            except OSError:
                pass
            raise
        temp.close()
        return temp.name, filename, current_url
    finally:
        response.close()
        response._deepdetect_connection.close()


def _download_public_platform_video(url: str, max_bytes: int):
    """Resolve a supported public video page without bypassing access controls."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise ValueError(
            "This is a video page rather than a direct file URL. "
            "Install the URL-import dependencies and try again."
        ) from exc

    temp_dir = tempfile.mkdtemp(prefix="deepdetect_url_")
    output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
    options = {
        "format": "best[ext=mp4][vcodec^=avc1][acodec^=mp4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "max_filesize": max_bytes,
        "socket_timeout": 20,
        "retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            downloaded = Path(downloader.prepare_filename(info))
            candidates = list(Path(temp_dir).glob("*"))
            if not downloaded.is_file() and candidates:
                downloaded = candidates[0]
            if not downloaded.is_file() or downloaded.stat().st_size == 0:
                raise ValueError("The public page did not provide a downloadable video.")
            if downloaded.stat().st_size > max_bytes:
                raise ValueError(f"The video exceeds the {max_bytes // (1024 * 1024)} MB import limit.")
            filename = Path(info.get("title") or downloaded.name).stem + downloaded.suffix
            return str(downloaded), filename, url
    except ValueError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("login", "sign in", "private", "drm", "authentication")):
            raise ValueError(
                "This video is private, login-protected, DRM-protected, or requires authentication."
            ) from exc
        raise ValueError(
            "The public video page could not be downloaded. Use a direct video URL "
            "or upload the video file."
        ) from exc
