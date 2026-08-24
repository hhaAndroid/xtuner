from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("domino_ep_event_timeline.png")
W, H = 1800, 1120


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_HEAD = font(24, True)
F = font(20)
F_SMALL = font(16)
F_TINY = font(14)

BG = (250, 252, 255)
INK = (32, 40, 56)
MUTED = (100, 112, 132)
GRID = (224, 230, 240)
COMPUTE = (217, 232, 255)
COMPUTE_BORDER = (55, 112, 210)
COMM = (216, 242, 226)
COMM_BORDER = (44, 145, 88)
WAIT = (255, 239, 210)
WAIT_BORDER = (204, 133, 32)
EVENT = (242, 95, 76)
ARROW = (58, 73, 98)


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, fill=INK, f=F) -> None:
    bbox = draw.textbbox((0, 0), label, font=f)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2 - 1
    draw.text((x, y), label, fill=fill, font=f)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, fill, outline, f=F) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)
    text_center(draw, box, label, f=f)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=ARROW, width=2) -> None:
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) > abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - 10 * sign, y2 - 5), (x2 - 10 * sign, y2 + 5)]
    else:
        sign = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 5, y2 - 10 * sign), (x2 + 5, y2 - 10 * sign)]
    draw.polygon(pts, fill=fill)


def event_dot(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, above: bool = True) -> None:
    r = 6
    draw.ellipse((x - r, y - r, x + r, y + r), fill=EVENT, outline=(150, 42, 34), width=1)
    bbox = draw.textbbox((0, 0), label, font=F_TINY)
    tx = x - (bbox[2] - bbox[0]) / 2
    ty = y - 24 if above else y + 10
    draw.text((tx, ty), label, fill=EVENT, font=F_TINY)


def stream_line(draw: ImageDraw.ImageDraw, y: int, x0: int, x1: int, label: str) -> None:
    draw.line((x0, y, x1, y), fill=GRID, width=3)
    draw.text((36, y - 13), label, fill=INK, font=F)


def panel_header(draw: ImageDraw.ImageDraw, y: int, title: str) -> None:
    draw.text((40, y), title, fill=INK, font=F_HEAD)
    draw.line((40, y + 38, W - 40, y + 38), fill=GRID, width=2)


def draw_forward(draw: ImageDraw.ImageDraw) -> None:
    panel_header(draw, 90, "Forward: compute stream and comm stream are synchronized by forward events")
    yc, ym = 220, 350
    x0, x1 = 150, 1660
    stream_line(draw, yc, x0, x1, "compute")
    stream_line(draw, ym, x0, x1, "comm")

    # Compute tasks.
    rounded(draw, (170, yc - 30, 300, yc + 30), "Pre0", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (330, yc - 30, 460, yc + 30), "Pre1", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (500, yc - 24, 545, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (560, yc - 30, 690, yc + 30), "Expert0", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (730, yc - 24, 775, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (790, yc - 30, 920, yc + 30), "Expert1", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (960, yc - 24, 1005, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (1020, yc - 30, 1140, yc + 30), "Post0", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (1180, yc - 24, 1225, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (1240, yc - 30, 1360, yc + 30), "Post1", COMPUTE, COMPUTE_BORDER)

    # Comm tasks, ordered on a single communication stream.
    rounded(draw, (310, ym - 30, 500, ym + 30), "Dispatch0", COMM, COMM_BORDER)
    rounded(draw, (510, ym - 30, 700, ym + 30), "Dispatch1", COMM, COMM_BORDER)
    rounded(draw, (710, ym - 30, 900, ym + 30), "Combine0", COMM, COMM_BORDER)
    rounded(draw, (910, ym - 30, 1140, ym + 30), "Combine1", COMM, COMM_BORDER)

    # forward_previous_event: compute records, comm waits.
    for name, cx, mx in [
        ("D0.p", 300, 310),
        ("D1.p", 460, 510),
        ("C0.p", 690, 710),
        ("C1.p", 920, 910),
    ]:
        event_dot(draw, cx, yc, f"rec {name}", above=True)
        arrow(draw, (cx, yc + 35), (mx, ym - 35))
        event_dot(draw, mx, ym, f"wait {name}", above=False)

    # forward_finished_event: comm records, compute waits.
    for name, mx, cx in [
        ("D0.d", 500, 500),
        ("D1.d", 700, 730),
        ("C0.d", 900, 960),
        ("C1.d", 1140, 1180),
    ]:
        event_dot(draw, mx, ym, f"rec {name}", above=True)
        arrow(draw, (mx, ym - 35), (cx, yc + 35))
        event_dot(draw, cx, yc, f"wait {name}", above=False)


def draw_backward(draw: ImageDraw.ImageDraw) -> None:
    panel_header(draw, 505, "Backward: hooks record grad-ready events; prehooks wait for comm-done events")
    yc, ym = 650, 780
    x0, x1 = 150, 1660
    stream_line(draw, yc, x0, x1, "compute")
    stream_line(draw, ym, x0, x1, "comm")

    # Compute tasks. The key visual target is C1 reverse A2A overlapping Post0 backward.
    rounded(draw, (170, yc - 30, 300, yc + 30), "Post1 bw", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (330, yc - 30, 560, yc + 30), "Post0 bw", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (580, yc - 24, 625, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (640, yc - 30, 800, yc + 30), "Expert1 bw", COMPUTE, COMPUTE_BORDER)
    rounded(draw, (820, yc - 24, 865, yc + 24), "wait", WAIT, WAIT_BORDER, F_SMALL)
    rounded(draw, (880, yc - 30, 1040, yc + 30), "Expert0 bw", COMPUTE, COMPUTE_BORDER)

    # Reverse comm tasks.
    rounded(draw, (310, ym - 30, 540, ym + 30), "Combine1 bw A2A", COMM, COMM_BORDER)
    rounded(draw, (570, ym - 30, 780, ym + 30), "Combine0 bw A2A", COMM, COMM_BORDER)

    # C1 backward_previous_event / backward_finished_event.
    event_dot(draw, 300, yc, "rec C1.p", above=True)
    arrow(draw, (300, yc + 35), (310, ym - 35))
    event_dot(draw, 310, ym, "wait C1.p", above=False)

    event_dot(draw, 540, ym, "rec C1.d", above=True)
    arrow(draw, (540, ym - 35), (580, yc + 35))
    event_dot(draw, 580, yc, "wait C1.d", above=False)

    # C0 backward_previous_event / backward_finished_event.
    event_dot(draw, 560, yc, "rec C0.p", above=True)
    arrow(draw, (560, yc + 35), (570, ym - 35))
    event_dot(draw, 570, ym, "wait C0.p", above=False)

    event_dot(draw, 780, ym, "rec C0.d", above=True)
    arrow(draw, (780, ym - 35), (820, yc + 35))
    event_dot(draw, 820, yc, "wait C0.d", above=False)

    # Emphasize overlap regions.
    draw.rounded_rectangle((320, 835, 555, 875), radius=8, fill=(255, 248, 228), outline=(214, 160, 56), width=1)
    draw.text((335, 844), "C1 bw A2A overlaps Post0 bw", fill=(120, 80, 20), font=F_SMALL)
    draw.line((330, yc + 44, 540, yc + 44), fill=(214, 160, 56), width=4)
    draw.line((310, ym - 44, 540, ym - 44), fill=(214, 160, 56), width=4)

    draw.rounded_rectangle((570, 888, 815, 928), radius=8, fill=(255, 248, 228), outline=(214, 160, 56), width=1)
    draw.text((585, 897), "C0 bw A2A overlaps Expert1 bw", fill=(120, 80, 20), font=F_SMALL)
    draw.line((640, yc + 48, 780, yc + 48), fill=(214, 160, 56), width=4)
    draw.line((570, ym - 48, 780, ym - 48), fill=(214, 160, 56), width=4)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    y = 990
    draw.text((40, y), "Legend", fill=INK, font=F_HEAD)
    rounded(draw, (150, y - 5, 285, y + 40), "compute", COMPUTE, COMPUTE_BORDER, F_SMALL)
    rounded(draw, (310, y - 5, 445, y + 40), "comm", COMM, COMM_BORDER, F_SMALL)
    rounded(draw, (470, y - 5, 560, y + 40), "wait", WAIT, WAIT_BORDER, F_SMALL)
    event_dot(draw, 610, y + 17, "event", above=False)
    draw.text(
        (700, y - 2),
        "rec = record event, p = previous/ready event, d = done/finished event.",
        fill=MUTED,
        font=F_SMALL,
    )
    draw.text(
        (700, y + 25),
        "record_stream is a tensor-storage lifetime marker, not a stream synchronization event.",
        fill=MUTED,
        font=F_SMALL,
    )


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((40, 28), "XTuner Event/Comm-Stream Timeline for Domino EP", fill=INK, font=F_TITLE)
    draw.text(
        (40, 66),
        "Two CUDA streams: compute stream runs model kernels; comm stream runs all-to-all. CUDA events connect them.",
        fill=MUTED,
        font=F,
    )

    draw_forward(draw)
    draw_backward(draw)
    draw_legend(draw)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
