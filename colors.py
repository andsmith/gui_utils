

RGB_COLORS = dict(
    # for UI
    black=(8, 8, 8),
    white=(255, 255, 255),
    off_white=(252, 246, 232),
    gray=(128, 128, 128),
    dark_gray=(64, 64, 64),
    dark_dark_gray=(32, 32, 32),
    light_gray=(128 + 64, 128 + 64, 128 + 64),
    light_light_gray=(128 + 64 + 32, 128 + 64 + 32, 128 + 64 + 32),
    slightly_light_gray=(128 + 16, 128 + 16, 128 + 16),
    off_black=(16, 16, 16),
    light_green=(64, 255, 64),
    light_red=(255, 64, 64),
    light_cyan=(10, 255, 255),
    pencil_yellow=(255, 182, 5, 255),
    eraser_pink=(198, 82, 122, 255),
    steel=(136, 139, 141),
    wood=(205, 170, 125),
    smurf=(70, 149, 214),
    sky=(25, 189, 255),

    # for drawing
    red=(255, 0, 0),
    orange=(255, 127, 0),
    yellow=(255, 255, 0),
    green=(0, 255, 0),
    blue=(0, 0, 255),
    brown=(165, 42, 42),
    purple=(100, 0, 235))

def add_alpha(color, alpha):
    return color[0], color[1], color[2], alpha



RGBA_COLORS_OPAQUE = {k: add_alpha(RGB_COLORS[k], 255) for k in RGB_COLORS}