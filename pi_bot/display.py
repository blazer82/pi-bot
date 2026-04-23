"""FT81x display output (optional, gracefully skipped if unavailable)."""

try:
    from ft81x import FT81x
except ImportError:
    FT81x = None

_display = None
_last_user = ""

_WIDTH = 480
_HEIGHT = 480
_BG = 0x000000
_USER_COLOR = 0x888888
_BOT_COLOR = 0xFFFFFF
_STATUS_COLOR = 0xFFFFFF
_USER_FONT = 28
_BOT_FONT = 30
_STATUS_FONT = 31
_MARGIN = 20
_USER_ZONE_Y = 20
_BOT_ZONE_Y = 140
_LINE_H_28 = 28
_LINE_H_30 = 32
_CHARS_28 = 27
_CHARS_30 = 23


def _wrap(text, chars_per_line):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > chars_per_line:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def _render_conversation(user_text, bot_text):
    _display.begin_display_list()
    _display.clear(_BG)

    y = _USER_ZONE_Y
    _display.draw_text(_MARGIN, y, _USER_FONT, _USER_COLOR, 0, "Du:")
    y += _LINE_H_28
    for line in _wrap(user_text, _CHARS_28):
        if y > _BOT_ZONE_Y - _LINE_H_28:
            break
        _display.draw_text(_MARGIN, y, _USER_FONT, _USER_COLOR, 0, line)
        y += _LINE_H_28

    y = _BOT_ZONE_Y
    _display.draw_text(_MARGIN, y, _BOT_FONT, _BOT_COLOR, 0, "Pi-Bot:")
    y += _LINE_H_30
    for line in _wrap(bot_text, _CHARS_30):
        if y > _HEIGHT - _MARGIN:
            break
        _display.draw_text(_MARGIN, y, _BOT_FONT, _BOT_COLOR, 0, line)
        y += _LINE_H_30

    _display.swap()


def _render_status(text):
    _display.begin_display_list()
    _display.clear(_BG)
    _display.draw_text(
        _WIDTH // 2, _HEIGHT // 2, _STATUS_FONT, _STATUS_COLOR,
        FT81x.OPT_CENTER, text,
    )
    _display.swap()


def init_display():
    global _display
    if FT81x is None:
        print("Display: ft81x not installed, skipping.")
        return
    try:
        _display = FT81x()
        _render_status("Pi Bot")
        print("Display: initialized.")
    except Exception as e:
        print(f"Display: init failed ({e}), skipping.")
        _display = None


def show_ready():
    if _display is None:
        return
    _render_status("Pi Bot ist bereit.")


def show_listening():
    if _display is None:
        return
    _render_status("Ich hoere zu...")


def show_thinking():
    if _display is None:
        return
    _render_status("Ich denke nach...")


def show_user_text(text):
    global _last_user
    if _display is None:
        return
    _last_user = text
    _render_conversation(text, "")


def show_bot_text(text):
    if _display is None:
        return
    _render_conversation(_last_user, text)


def close_display():
    global _display
    if _display is None:
        return
    try:
        _display.close()
    except Exception:
        pass
    _display = None
