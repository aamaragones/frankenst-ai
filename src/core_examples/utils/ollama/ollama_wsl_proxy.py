import os
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

WINDOWS_CURL = os.getenv("WINDOWS_CURL_PATH", "/mnt/c/WINDOWS/system32/curl.exe")
TARGET_BASE_URL = os.getenv("OLLAMA_WINDOWS_BASE_URL", "http://127.0.0.1:11434")
PROXY_HOST = os.getenv("OLLAMA_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("OLLAMA_PROXY_PORT", "11435"))


def _is_wsl() -> bool:
    if os.getenv("WSL_INTEROP") or os.getenv("WSL_DISTRO_NAME"):
        return True

    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _can_connect(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_wsl_windows_hosts() -> list[str]:
    hosts: list[str] = []

    try:
        for line in Path("/etc/resolv.conf").read_text().splitlines():
            if line.startswith("nameserver "):
                hosts.append(line.split()[1])
                break
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["ip", "route"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("default via "):
                hosts.append(line.split()[2])
                break
    except OSError:
        pass

    unique_hosts: list[str] = []
    for host in hosts:
        if host not in unique_hosts:
            unique_hosts.append(host)

    return unique_hosts


def _has_windows_curl_access_to_ollama(target_base_url: str) -> bool:
    windows_curl = Path(WINDOWS_CURL)
    if not windows_curl.exists():
        return False

    try:
        result = subprocess.run(
            [str(windows_curl), "-sS", "--max-time", "2", f"{target_base_url}/api/tags"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False

    return result.returncode == 0


def _ensure_wsl_ollama_proxy(proxy_host: str, proxy_port: int, target_base_url: str) -> str | None:
    if _can_connect(proxy_host, proxy_port):
        return f"http://{proxy_host}:{proxy_port}"

    proxy_script = Path(__file__)
    process_env = os.environ.copy()
    process_env.setdefault("OLLAMA_PROXY_HOST", proxy_host)
    process_env.setdefault("OLLAMA_PROXY_PORT", str(proxy_port))
    process_env.setdefault("OLLAMA_WINDOWS_BASE_URL", target_base_url)

    try:
        subprocess.Popen(
            [sys.executable, str(proxy_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=process_env,
        )
    except OSError:
        return None

    for _ in range(30):
        if _can_connect(proxy_host, proxy_port):
            return f"http://{proxy_host}:{proxy_port}"
        time.sleep(0.1)

    return None


def resolve_ollama_base_url(config_host: str | None = None, default_host: str = "http://127.0.0.1:11434", proxy_port: int = 11435) -> str:
    explicit_host = os.getenv("OLLAMA_HOST")
    if explicit_host:
        return explicit_host

    if config_host:
        return config_host

    parsed_default = urlparse(default_host)
    default_hostname = parsed_default.hostname or "127.0.0.1"
    default_port = parsed_default.port or 11434

    if _can_connect(default_hostname, default_port):
        return default_host

    if _is_wsl():
        for windows_host in _get_wsl_windows_hosts():
            if _can_connect(windows_host, default_port):
                return f"http://{windows_host}:{default_port}"

        if _has_windows_curl_access_to_ollama(default_host):
            proxy_url = _ensure_wsl_ollama_proxy(PROXY_HOST, proxy_port, default_host)
            if proxy_url:
                return proxy_url

    return default_host


class OllamaWSLProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy_request(self) -> None:
        self.close_connection = True
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""

        command = [
            WINDOWS_CURL,
            "-sS",
            "-N",
            "-i",
            "--http1.1",
            "-X",
            self.command,
            f"{TARGET_BASE_URL}{self.path}",
        ]

        for header_name, header_value in self.headers.items():
            if header_name.lower() in {"host", "content-length", "connection", "accept-encoding"}:
                continue
            command.extend(["-H", f"{header_name}: {header_value}"])

        if body:
            command.extend(["--data-binary", "@-"])

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if body else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if body and process.stdin is not None:
            process.stdin.write(body)
            process.stdin.close()

        raw_response = bytearray()
        while b"\r\n\r\n" not in raw_response:
            chunk = process.stdout.read(1)
            if not chunk:
                break
            raw_response.extend(chunk)
            if len(raw_response) > 65536:
                break

        header_blob, separator, remainder = raw_response.partition(b"\r\n\r\n")
        if not separator:
            stderr_output = process.stderr.read().decode("utf-8", errors="replace")
            self.send_error(502, stderr_output or "Invalid response from Windows Ollama relay")
            process.kill()
            return

        header_lines = header_blob.decode("iso-8859-1").split("\r\n")
        status_line = header_lines[0]
        status_parts = status_line.split(" ", 2)
        status_code = int(status_parts[1]) if len(status_parts) > 1 else 502
        status_message = status_parts[2] if len(status_parts) > 2 else ""

        self.send_response(status_code, status_message)
        for header_line in header_lines[1:]:
            if not header_line or ":" not in header_line:
                continue
            header_name, header_value = header_line.split(":", 1)
            if header_name.lower() in {"transfer-encoding", "content-length", "connection", "keep-alive", "proxy-connection"}:
                continue
            self.send_header(header_name, header_value.strip())
        self.send_header("Connection", "close")
        self.end_headers()

        if remainder:
            self.wfile.write(remainder)
            self.wfile.flush()

        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except BrokenPipeError:
                process.kill()
                return

        process.wait(timeout=5)

    def do_GET(self) -> None:
        self._proxy_request()

    def do_POST(self) -> None:
        self._proxy_request()

    def do_PUT(self) -> None:
        self._proxy_request()

    def do_DELETE(self) -> None:
        self._proxy_request()

    def do_HEAD(self) -> None:
        self._proxy_request()

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((PROXY_HOST, PROXY_PORT), OllamaWSLProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()