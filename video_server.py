#!/usr/bin/env python3
"""
HTTP Range Request 対応の動画配信サーバー
通常の python -m http.server は Range Request に非対応 → シーク不可
このスクリプトは 206 Partial Content を正しく返す
"""
import http.server
import os
import sys


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Range Request (RFC 7233) に対応したファイルサーバー"""

    def do_GET(self):
        f = self.send_head()
        if f:
            try:
                if getattr(self, '_range_length', None) is not None:
                    # Range request: 指定バイト数だけ送信
                    remaining = self._range_length
                    while remaining > 0:
                        chunk = min(65536, remaining)
                        data = f.read(chunk)
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
                    self._range_length = None
                else:
                    # 通常リクエスト: ファイル全体を送信
                    self.copyfile(f, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass  # クライアントが切断した場合は無視
            finally:
                f.close()

    def send_head(self):
        path = self.translate_path(self.path)
        ctype = self.guess_type(path)
        self._range_length = None

        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            file_size = fs[6]
        except Exception:
            f.close()
            self.send_error(500)
            return None

        range_header = self.headers.get('Range', '')

        if range_header.startswith('bytes='):
            try:
                spec = range_header[6:].split('-')
                start = int(spec[0]) if spec[0] else 0
                end   = int(spec[1]) if len(spec) > 1 and spec[1] else file_size - 1
                end   = min(end, file_size - 1)

                if start > end or start >= file_size:
                    self.send_error(416, "Range Not Satisfiable")
                    f.close()
                    return None

                length = end - start + 1
                f.seek(start)
                self._range_length = length

                self.send_response(206, "Partial Content")
                self.send_header('Content-Type', ctype)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(length))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                return f

            except Exception as e:
                f.close()
                self.send_error(400, f"Bad Range: {e}")
                return None

        # 通常リクエスト
        self.send_response(200, "OK")
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(file_size))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        return f

    def log_message(self, format, *args):
        pass  # アクセスログを抑制


if __name__ == '__main__':
    port      = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2]      if len(sys.argv) > 2 else 'static'
    directory = os.path.abspath(directory)

    handler_cls = lambda *args, **kwargs: RangeHTTPRequestHandler(
        *args, directory=directory, **kwargs
    )

    with http.server.HTTPServer(('', port), handler_cls) as httpd:
        print(f"✅ Range-capable server: http://localhost:{port}/  (dir: {directory})")
        httpd.serve_forever()
