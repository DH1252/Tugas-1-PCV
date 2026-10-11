import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (90, 220, 120)
RED = (70, 70, 230)
YELLOW = (60, 220, 255)
HAND_COLORS = ((255, 200, 80), (120, 120, 255))
AA = cv2.LINE_AA


def text(img, s, org, scale=0.7, color=WHITE, thick=2, center=False):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if center:
        x -= tw // 2
    off = max(1, thick)
    cv2.putText(img, s, (x + off, y + off), FONT, scale, (15, 15, 15), thick, AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, AA)
    return tw, th


def dim(frame, factor):
    return cv2.convertScaleAbs(frame, alpha=factor)


def blend(img, overlay, alpha):
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def hsv_color(h, s=190, v=255):
    px = np.uint8([[[h % 180, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def _star(cx, cy, r, angle, points=5):
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * 0.45
        a = angle + i * np.pi / points
        pts.append((cx + rad * np.sin(a), cy - rad * np.cos(a)))
    return np.array(pts, np.int32)


def draw_item(img, item, t):
    x, y, r = int(item.pos[0]), int(item.pos[1]), int(item.radius)
    if item.kind == "ball":
        cv2.circle(img, (x, y), r, hsv_color(item.hue), -1, AA)
        cv2.circle(img, (x - r // 3, y - r // 3), max(2, r // 4), (245, 245, 245), -1, AA)
        cv2.circle(img, (x, y), r, (30, 30, 30), 2, AA)
    elif item.kind == "balloon":
        color = hsv_color(item.hue, 160)
        cv2.line(img, (x, y + int(1.2 * r)), (x + 4, y + int(2.4 * r)), GREY, 1, AA)
        cv2.ellipse(img, (x, y), (r, int(1.2 * r)), 0, 0, 360, color, -1, AA)
        knot = np.array([(x, y + int(1.2 * r)), (x - 5, y + int(1.2 * r) + 7), (x + 5, y + int(1.2 * r) + 7)], np.int32)
        cv2.fillPoly(img, [knot], color, AA)
        cv2.ellipse(img, (x - r // 3, y - r // 2), (max(2, r // 5), max(3, r // 3)), 20, 0, 360, (240, 240, 240), -1, AA)
    elif item.kind == "heavy":
        cv2.circle(img, (x, y), r, (60, 60, 60), -1, AA)
        cv2.circle(img, (x, y), r - 5, (95, 95, 95), 2, AA)
        cv2.circle(img, (x - r // 3, y - r // 3), max(2, r // 5), (200, 200, 200), -1, AA)
        text(img, "kg", (x, y + 6), 0.5, WHITE, 1, center=True)
    elif item.kind == "spike":
        cv2.fillPoly(img, [_star(x, y, r * 1.35, t * 3, 8)], (40, 40, 200), AA)
        cv2.circle(img, (x, y), int(r * 0.7), (30, 30, 150), -1, AA)
    elif item.kind == "star":
        cv2.fillPoly(img, [_star(x, y, r * 1.2, t * 2)], YELLOW, AA)
        cv2.polylines(img, [_star(x, y, r * 1.2, t * 2)], True, (20, 120, 200), 2, AA)


def paddle_fill(overlay, hands):
    for h in hands:
        color = HAND_COLORS[h.id % len(HAND_COLORS)]
        cv2.circle(overlay, (int(h.palm[0]), int(h.palm[1])), int(h.radius), color, -1, AA)


def paddle_outline(img, hands, debug=False):
    for h in hands:
        color = HAND_COLORS[h.id % len(HAND_COLORS)]
        c = (int(h.palm[0]), int(h.palm[1]))
        cv2.circle(img, c, int(h.radius), color if h.found else GREY, 3 if h.found else 1, AA)
        cv2.circle(img, c, 4, color, -1, AA)
        if debug and h.contour is not None:
            cv2.drawContours(img, [h.contour], -1, color, 1, AA)


def draw_button(img, button, highlight=False):
    x, y, w, h = button.rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), YELLOW if highlight or button.progress > 0 else WHITE, 2, AA)
    text(img, button.label, (x + w // 2, y + h // 2 + 9), 0.8, WHITE, 2, center=True)
    if button.progress > 0:
        draw_ring(img, (x + w - 22, y + h // 2), 14, button.progress)


def draw_ring(img, center, radius, progress, color=GREEN):
    cv2.ellipse(img, (int(center[0]), int(center[1])), (radius, radius), -90, 0, 360 * min(progress, 1.0), color, 4, AA)


def hearts(img, lives, max_lives, org):
    for i in range(max_lives):
        c = (org[0] + i * 26, org[1])
        color = (80, 80, 230) if i < lives else (80, 80, 80)
        cv2.circle(img, (c[0] - 5, c[1] - 4), 7, color, -1, AA)
        cv2.circle(img, (c[0] + 5, c[1] - 4), 7, color, -1, AA)
        cv2.fillPoly(img, [np.array([(c[0] - 12, c[1] - 1), (c[0] + 12, c[1] - 1), (c[0], c[1] + 12)], np.int32)], color, AA)


def confidence_bars(img, hands, org):
    x, y = org
    for i, h in enumerate(hands):
        color = HAND_COLORS[h.id % len(HAND_COLORS)]
        yy = y + i * 18
        cv2.rectangle(img, (x, yy), (x + 100, yy + 10), (60, 60, 60), -1)
        cv2.rectangle(img, (x, yy), (x + int(100 * np.clip(h.confidence, 0, 1)), yy + 10), color, -1)
        text(img, f"hand {h.id}", (x - 70, yy + 10), 0.4, WHITE, 1)


def tile(img, size):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def debug_grid(views, timings, info="", size=(320, 240), cols=3):
    if not views:
        return None
    tiles = []
    for name, img in views.items():
        t = tile(img, size)
        text(t, name, (8, 22), 0.6, YELLOW, 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((size[1], size[0], 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    bar = np.zeros((52 if info else 28, grid.shape[1], 3), np.uint8)
    text(bar, "  ".join(f"{k} {v:.1f}ms" for k, v in timings.items()), (8, 20), 0.5, WHITE, 1)
    if info:
        text(bar, info, (8, 44), 0.5, YELLOW, 1)
    return np.vstack([grid, bar])
