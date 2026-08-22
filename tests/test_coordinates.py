from app.android.device import normalized_to_pixels


def test_normalized_coordinates_map_to_screenshot_pixels() -> None:
    assert normalized_to_pixels(0, 0, 1080, 2400) == (0, 0)
    assert normalized_to_pixels(500, 500, 1080, 2400) == (540, 1200)
    assert normalized_to_pixels(1000, 1000, 1080, 2400) == (1080, 2400)


def test_coordinate_conversion_uses_each_axis_dimension() -> None:
    assert normalized_to_pixels(250, 750, 800, 1200) == (200, 900)
