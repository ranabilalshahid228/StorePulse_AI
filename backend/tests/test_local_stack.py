from __future__ import annotations

from scripts.local_stack import TUNNEL_URL, rank_adapters


def test_wifi_preferred_and_vpn_skipped() -> None:
    found = [
        ("vpn.oth-regensburg.de", "172.25.32.73"),
        ("Ethernet", "192.168.1.2"),
        ("WLAN", "10.62.37.90"),
        ("vEthernet (WSL)", "172.20.0.1"),
        ("Loopback Pseudo-Interface 1", "127.0.0.1"),
        ("Wi-Fi 2", "169.254.3.4"),
    ]
    assert rank_adapters(found) == [("WLAN", "10.62.37.90"), ("Ethernet", "192.168.1.2")]


def test_tunnel_url_is_found_in_cloudflared_output() -> None:
    line = "INF |  https://quiet-river-1234.trycloudflare.com                |"
    match = TUNNEL_URL.search(line)
    assert match and match.group(0) == "https://quiet-river-1234.trycloudflare.com"
