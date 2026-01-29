#!/usr/bin/env python3
"""
Simple HTTP server with Range request support for testing.
"""

import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with Range request support."""

    def send_head(self):
        """Handle HEAD and GET requests with Range support."""
        path = self.translate_path(self.path)

        if os.path.isdir(path):
            return super().send_head()

        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get('Range')

        if range_header:
            # Parse Range header
            match = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                end = min(end, file_size - 1)

                if start >= file_size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return None

                content_length = end - start + 1

                self.send_response(206)
                self.send_header('Content-Type', self.guess_type(path))
                self.send_header('Content-Length', content_length)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                return open(path, 'rb'), start, content_length

        # No Range header - serve full file
        self.send_response(200)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Length', file_size)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        return open(path, 'rb'), 0, file_size

    def do_GET(self):
        result = self.send_head()
        if result:
            f, start, length = result
            try:
                f.seek(start)
                self.wfile.write(f.read(length))
            finally:
                f.close()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.end_headers()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HTTP server with Range support')
    parser.add_argument('-p', '--port', type=int, default=8080)
    parser.add_argument('-d', '--directory', default='.')
    args = parser.parse_args()

    os.chdir(args.directory)

    server = HTTPServer(('', args.port), RangeRequestHandler)
    print(f"Serving {args.directory} on http://localhost:{args.port}")
    print("Range requests enabled. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == '__main__':
    main()
