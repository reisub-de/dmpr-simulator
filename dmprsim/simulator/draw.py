import math
import shutil
from pathlib import Path

try:
    import cairo
except ImportError:
    import cairocffi as cairo

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def color_links_light(index):
    return .1, .1, .1, .4


def color_links_dark(index):
    table = ((1.0, 0.15, 0.15, 1.0), (0.15, 1.0, 0.15, 1.0),
             (0.45, 0.2, 0.15, 1.0), (0.85, 0.5, 0.45, 1.0))
    return table[index]


def color_links(args, index):
    if args.color_scheme == "light":
        return color_links_light(index)
    else:
        return color_links_dark(index)


def color_db_light():
    return 1.000, 1.000, 1.000, 1.0


def color_db_dark():
    return 0.15, 0.15, 0.15, 1.0


def color_db(args):
    if args.color_scheme == "light":
        return color_db_light()
    else:
        return color_db_dark()


def color_range_light(index):
    return 1.0, 1.0, 1.0, 0.1


def color_range_dark(index):
    color = ((1.0, 1.0, 0.5, 0.05), (1.0, 0.0, 1.0, 0.05),
             (0.5, 1.0, 0.0, 0.05), (1.0, 0.5, 1.0, 0.05))
    return color[index]


def color_range(args, index):
    if args.color_scheme == "light":
        return color_range_light(index)
    else:
        return color_range_dark(index)


def color_node_inner_light():
    return 0.1, 0.1, 0.1


def color_node_inner_dark():
    return 0.5, 1, 0.7


def color_node_inner(args):
    if args.color_scheme == "light":
        return color_node_inner_light()
    else:
        return color_node_inner_dark()


def color_node_transmitter_outline_light():
    return 1.0, 0.0, 1.0


def color_node_transmitter_outline_dark():
    return 0.3, 0.5, 0.0


def color_node_transmitter_outline(args):
    if args.color_scheme == "light":
        return color_node_transmitter_outline_light()
    else:
        return color_node_transmitter_outline_dark()


def color_node_receiver_outline_light():
    return 0.9, .2, .1


def color_node_receiver_outline_dark():
    return 1.0, 0.0, 1.0


def color_node_receiver_outline(args):
    if args.color_scheme == "light":
        return color_node_receiver_outline_light()
    else:
        return color_node_receiver_outline_dark()


def color_text_light():
    return 0., 0., 0.


def color_text_dark():
    return 0.5, 1, 0.7


def color_text(args):
    if args.color_scheme == "light":
        return color_text_light()
    else:
        return color_text_dark()


def color_text_inverse_light():
    return 1., 1., 1.


def color_text_inverse_dark():
    return 1, 0, 0


def color_text_inverse(args):
    if args.color_scheme == "light":
        return color_text_inverse_light()
    else:
        return color_text_inverse_dark()


def draw_router_loc(args, ld: Path, area, r, ctx: cairo.Context):
    ctx.rectangle(0, 0, area.x, area.y)
    ctx.set_source_rgba(*color_db(args))
    ctx.fill()

    for router in r:
        if not router.mm.visible:
            continue
        x, y = router.coordinates()

        ctx.set_line_width(0.1)
        path_thickness = 6.0
        # iterate over links
        interfaces_idx = 0
        for i, interface_name in enumerate(router.interfaces):
            range_ = router.interfaces[interface_name]['range']
            ctx.set_source_rgba(*color_range(args, i))
            ctx.move_to(x, y)
            ctx.arc(x, y, range_, 0, 2 * math.pi)
            ctx.fill()

            # draw lines between links
            ctx.set_line_width(path_thickness)
            for r_id, r_obj in router.connections[interface_name].items():
                if not r_obj.mm.visible:
                    continue
                other_x, other_y = r_obj.coordinates()
                ctx.move_to(x, y)
                ctx.set_source_rgba(*color_links(args, interfaces_idx))
                ctx.line_to(other_x, other_y)
                ctx.stroke()
            interfaces_idx += 1
            path_thickness -= 4.0
            if path_thickness < 2.0:
                path_thickness = 2.0

    # draw active links
    ctx.set_line_width(4.0)
    dash_len = 4.0
    for router in r:
        for tos, src, dst in router.forwarded_packets:
            ctx.set_source_rgba(1., 0, 0, 1)
            ctx.set_dash([dash_len, dash_len])
            if tos == "lowest-loss":
                ctx.set_source_rgba(0., 0., 1., 1)
                ctx.set_dash([dash_len, dash_len], dash_len)
            s_x, s_y = src.coordinates()
            d_x, d_y = dst.coordinates()
            ctx.move_to(s_x, s_y)
            ctx.line_to(d_x, d_y)
            ctx.stroke()
    # reset now
    for router in r:
        router.forwarded_packets = list()

    # draw text and node circle
    ctx.set_line_width(3.0)
    for router in r:
        if not router.mm.visible:
            continue
        x, y = router.coordinates()
        # node middle point
        ctx.move_to(x, y)
        ctx.arc(x, y, 5, 0, 2 * math.pi)

        if router.is_transmitter:
            ctx.set_source_rgb(*color_node_transmitter_outline(args))
            ctx.stroke_preserve()

            ctx.set_source_rgb(*color_text_inverse(args))
            ctx.move_to(x + 11, y - 9)
            ctx.set_antialias(False)
            ctx.show_text("Transmitter")
            ctx.set_antialias(True)

            ctx.set_source_rgb(*color_node_transmitter_outline(args))
            ctx.move_to(x + 10, y - 10)
            ctx.set_antialias(False)
            ctx.show_text("Transmitter")
            ctx.set_antialias(True)

        if router.is_receiver:
            # draw circle outline
            ctx.set_source_rgb(*color_node_receiver_outline(args))
            ctx.stroke_preserve()

            ctx.set_source_rgb(*color_text_inverse(args))
            ctx.move_to(x + 11, y - 9)
            ctx.set_antialias(False)
            ctx.show_text("Receiver")
            ctx.set_antialias(True)

            # show text
            ctx.set_source_rgb(*color_node_receiver_outline(args))
            ctx.move_to(x + 10, y - 10)
            ctx.set_antialias(False)
            ctx.show_text("Receiver")
            ctx.set_antialias(True)

        ctx.set_source_rgb(*color_node_inner(args))
        ctx.fill()
        # show router id, first we draw a background
        # with a little offset and the inverse color,
        # later we draw the actual text
        ctx.set_font_size(10)
        ctx.set_source_rgb(*color_text_inverse(args))
        ctx.move_to(x + 10 + 1, y + 10 + 1)
        ctx.show_text(str(router.id))
        ctx.set_source_rgb(*color_text(args))
        ctx.move_to(x + 10, y + 10)
        ctx.set_antialias(False)
        ctx.show_text(str(router.id))
        ctx.set_antialias(True)


def color_transmission_circle_light():
    return 0., .0, .0, .2


def color_transmission_circle_dark():
    return .10, .10, .10, 1.0


def color_transmission_circle(args):
    if args.color_scheme == "light":
        return color_transmission_circle_light()
    else:
        return color_transmission_circle_dark()


def color_tx_links_light():
    return .1, .1, .1, .4


def color_tx_links_dark():
    return .0, .0, .0, .4


def color_tx_links(args):
    if args.color_scheme == "light":
        return color_tx_links_light()
    else:
        return color_tx_links_dark()


def draw_router_transmission(args, ld: Path, area, r, ctx: cairo.Context):
    ctx.rectangle(0, 0, area.x, area.y)
    ctx.set_source_rgba(*color_db(args))
    ctx.fill()

    # transmitting circles
    for router in r:
        if not router.mm.visible:
            continue
        x, y = router.coordinates()

        if router.transmission_within_second:
            distance = \
                max(router.interfaces.values(), key=lambda x: x['range'])[
                    'range']
            ctx.set_source_rgba(*color_transmission_circle(args))
            ctx.move_to(x, y)
            ctx.arc(x, y, distance, 0, 2 * math.pi)
            ctx.fill()

    for router in r:
        if not router.mm.visible:
            continue
        x, y = router.coordinates()

        ctx.set_line_width(0.1)
        path_thickness = 6.0
        # iterate over links
        for i, interface_name in enumerate(router.interfaces):
            range_ = router.interfaces[interface_name]['range']

            # draw lines between links
            ctx.set_line_width(path_thickness)
            for r_id, r_obj in router.connections[interface_name].items():
                if not r_obj.mm.visible:
                    continue
                other_x, other_y = r_obj.coordinates()
                ctx.move_to(x, y)
                ctx.set_source_rgba(*color_tx_links(args))
                ctx.line_to(other_x, other_y)
                ctx.stroke()

            path_thickness -= 4.0
            if path_thickness < 2.0:
                path_thickness = 2.0

    # draw dots over all
    for router in r:
        if not router.mm.visible:
            continue
        x, y = router.coordinates()

        ctx.set_line_width(0.0)
        ctx.set_source_rgba(*color_tx_links(args))
        ctx.move_to(x, y)
        ctx.arc(x, y, 5, 0, 2 * math.pi)
        ctx.fill()


def draw_images(args, ld: Path, area, r, img_idx):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMAGE_WIDTH, IMAGE_HEIGHT)
    full_ctx = cairo.Context(surface)
    full_ctx.rectangle(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
    full_ctx.set_source_rgba(*color_db(args))
    full_ctx.fill()

    effective_width = IMAGE_WIDTH / 2
    scale_factor = min(effective_width / area.x, IMAGE_HEIGHT / area.y)

    x_center_offset = effective_width / 2 - (area.x * scale_factor / 2)
    y_center_offset = IMAGE_HEIGHT / 2 - (area.y * scale_factor / 2)

    # One context on the left, scaled to half the width and full height
    # and centered on x- and y-axis
    left_ctx = cairo.Context(surface)
    left_ctx.translate(x_center_offset, y_center_offset)
    left_ctx.scale(scale_factor, scale_factor)

    # The other context on the right, scaled to half the width and full height
    # and centered on the x- and y-axis on the right side
    right_ctx = cairo.Context(surface)
    right_ctx.translate(effective_width + x_center_offset, y_center_offset)
    right_ctx.scale(scale_factor, scale_factor)

    draw_router_transmission(args, ld, area, r, right_ctx)
    draw_router_loc(args, ld, area, r, left_ctx)

    surface.write_to_png(str(ld / 'images' / '{:05}.png'.format(img_idx)))


def setup_img_folder(log_directory: Path):
    path = log_directory / 'images'
    shutil.rmtree(str(path), ignore_errors=True)
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        pass
