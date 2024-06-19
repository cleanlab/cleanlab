"""
This script performs a spurious correlation test to detect and measure unintended associations
between different transformations of images and their labels in a synthetic dataset.

The process involves:
1. Generating images with various shapes (circles and squares).
2. Applying different filters (dark, blurry, and odd aspect ratio) to these images.
3. Creating datasets with these filtered images and comparing them to a standard set of images without filters.

The goal is to evaluate whether the application of specific filters results in lower correlation scores,
indicating that the transformations introduce spurious correlations that might not be present in the original data.
This helps ensure that these filters don't create misleading patterns that could affect the performance and reliability
of machine learning models trained on such data.

The test is implemented using the cleanlab library using Datalab module, which helps identify and quantify these spurious correlations.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random
from datasets import Dataset
import pytest
from cleanlab import Datalab

seed = 42
np.random.seed(seed=seed)


def create_base_image(size=(64, 64), background_color=(255, 255, 255)):
    """
    Creates a base image with the given size and background color.

    Args:
        size (tuple): The size of the image (width, height).
        background_color (tuple): The background color of the image (RGB).

    Returns:
        Image: The created base image.
    """
    return Image.new("RGB", size, background_color)


def draw_shape(draw, shape, color, offset_x, offset_y, shape_size):
    """
    Draws a specified shape on the image.

    Args:
        draw (ImageDraw): The drawing context.
        shape (str): The shape to draw ('circle' or 'square').
        color (tuple): The color of the shape (RGB).
        offset_x (int): The x offset of the shape.
        offset_y (int): The y offset of the shape.
        shape_size (int): The size of the shape.
    """
    if shape == "circle":
        draw.ellipse(
            [(offset_x, offset_y), (offset_x + shape_size, offset_y + shape_size)], fill=color
        )
    elif shape == "square":
        draw.rectangle(
            [(offset_x, offset_y), (offset_x + shape_size, offset_y + shape_size)], fill=color
        )


def add_noise(image):
    """
    Adds random noise to the image.

    Args:
        image (Image): The image to add noise to.

    Returns:
        Image: The image with added noise.
    """
    np_img = np.array(image)
    noise = np.random.normal(0, 0.5, np_img.shape).astype(np.uint8)
    np_img = np.clip(np_img + noise, 0, 255)
    return Image.fromarray(np_img)


def create_image(shape, color, size=(64, 64), background_color=(255, 255, 255)):
    """
    Creates an image with a given shape and color.

    Args:
        shape (str): The shape to draw ('circle' or 'square').
        color (tuple): The color of the shape (RGB).
        size (tuple): The size of the image (width, height).
        background_color (tuple): The background color of the image (RGB).

    Returns:
        Image: The generated image with the shape.
    """
    img = create_base_image(size, background_color)
    draw = ImageDraw.Draw(img)

    offset_x, offset_y, shape_size = randomize_shape_position_and_size(size)

    draw_shape(draw, shape, color, offset_x, offset_y, shape_size)

    img = add_noise(img)

    return img


def randomize_shape_position_and_size(size):
    """
    Randomizes the position and size of the shape.

    Args:
        size (tuple): The size of the image (width, height).

    Returns:
        tuple: The x offset, y offset, and size of the shape.
    """
    max_offset = 10
    offset_x = random.randint(0, max_offset)
    offset_y = random.randint(0, max_offset)
    shape_size = random.randint(20, size[0] - 20)

    return offset_x, offset_y, shape_size


# Transformations
def apply_dark(image):
    """Decreases brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(0.3)


def apply_blurry(image):
    """Applies Gaussian blur to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=5))


def apply_identity(image):
    """Returns the unchanged image."""
    return image


def apply_odd_aspect_ratio(image):
    """Changes the aspect ratio to make the image tall and skinny."""
    return image.resize((32, 128))


def generate_backgrounds(num_variations=5):
    """
    Generates a list of random background variations.

    Args:
        num_variations (int): The number of background variations to generate.

    Returns:
        list: A list of tuples with brightness and color.
    """
    backgrounds = []
    for _ in range(num_variations):
        brightness = random.uniform(0.2, 1.0)
        color = tuple(random.choices(range(256), k=3))
        backgrounds.append((brightness, color))
    return backgrounds


def apply_background(image, brightness, color):
    """
    Applies a background color to the image with specified brightness.

    Args:
        image (Image): The image to apply the background to.
        brightness (float): The brightness level of the background.
        color (tuple): The RGB color of the background.

    Returns:
        Image: The image with the background applied.
    """
    enhancer = ImageEnhance.Brightness(Image.new("RGB", image.size, color))
    background = enhancer.enhance(brightness)
    return Image.alpha_composite(background.convert("RGBA"), image.convert("RGBA")).convert("RGB")


def apply_filter(image, filter_function):
    """
    Applies a random filter from the filter functions to the image.

    Args:
        image (Image): The image to apply the filter to.
        filter_function (function): A filter function to apply.

    Returns:
        Image: The filtered image.
    """
    filtered_img = filter_function(image)
    return filtered_img


def generate_image_with_background(shape, color, filter_function, backgrounds):
    """
    Generates an image with a specified shape and color, applies a random filter,
    and then applies a random background.

    Args:
        shape (str): The shape to draw.
        color (tuple): The color of the shape.
        filter_function (function): A filter function to apply.
        backgrounds (list): A list of background variations.

    Returns:
        tuple: The generated image, filter type, brightness, and background color.
    """
    img = create_image(shape, color)
    filtered_img = apply_filter(img, filter_function)
    brightness, bg_color = random.choice(backgrounds)
    background_img = apply_background(filtered_img, brightness, bg_color)
    return background_img


def get_filter_functions():
    """
    Returns a dictionary of available filter functions.

    Returns:
        dict: A dictionary mapping filter names to filter functions.
    """
    filter_functions_map = {
        "dark": apply_dark,
        "blurry": apply_blurry,
        "identity": apply_identity,
        "odd_aspect_ratio": apply_odd_aspect_ratio,
    }
    return filter_functions_map


def get_filter_functions_without_identity():
    """
    Returns a dictionary of available filter functions, excluding 'identity'.

    Returns:
        dict: A dictionary mapping filter names to filter functions.
    """
    filter_functions_map = get_filter_functions()
    del filter_functions_map["identity"]
    return filter_functions_map


def generate_dataset(
    num_images_per_class=50,
    num_background_variations=5,
    circle_filter="identity",
    square_filter="identity",
):
    """
    Generates a toy dataset with images and corresponding labels.

    Args:
        num_images_per_class (int): The number of images per class.
        num_background_variations (int): The number of background variations.
        circle_filter (str): The filter to apply to circle images.
        square_filter (str): The filter to apply to square images.

    Returns:
        Dataset: The generated dataset.
    """
    shapes = ["circle", "square"]
    filter_functions = [circle_filter, square_filter]
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (192, 192, 192),  # Gray
    ]
    filter_functions_map = get_filter_functions()
    filter_functions = [filter_functions_map[circle_filter], filter_functions_map[square_filter]]
    backgrounds = generate_backgrounds(num_background_variations)

    data = []
    labels = []

    for shape, filter_function in zip(shapes, filter_functions):
        for _ in range(num_images_per_class):
            color = random.choice(colors)
            background_img = generate_image_with_background(
                shape, color, filter_function, backgrounds
            )

            data.append(background_img)
            labels.append(shape)

    dataset = Dataset.from_dict({"image": data, "label": labels})
    return dataset


def get_property_score(df, property):
    """
    Retrieves the score for a specific property from the dataframe.

    Args:
        df (DataFrame): The dataframe containing property scores.
        property (str): The property to retrieve the score for.

    Returns:
        float: The score for the specified property.
    """
    return df.loc[df["property"] == property, "score"].iloc[0]


def get_scores(df):
    """
    Retrieves scores for all relevant properties from the dataframe.

    Args:
        df (DataFrame): The dataframe containing property scores.

    Returns:
        dict: A dictionary with property names as keys and their scores as values.
    """
    filter_functions_map = get_filter_functions_without_identity()
    properties_of_interest = [prop + "_score" for prop in filter_functions_map.keys()]
    standard_correlation_scores = {
        prop: get_property_score(df, prop) for prop in properties_of_interest
    }
    return standard_correlation_scores


def get_correlation_scores(circle_filter="identity", square_filter="identity"):
    """
    Generates a dataset and computes spurious correlation scores.

    Args:
        circle_filter (str): The filter to apply to circle images.
        square_filter (str): The filter to apply to square images.

    Returns:
        dict: A dictionary with property names as keys and their correlation scores as values.
    """
    dataset = generate_dataset(circle_filter=circle_filter, square_filter=square_filter)
    lab = Datalab(data=dataset, label_name="label", image_key="image")
    lab.find_issues()
    correlation_scores = lab._spurious_correlation()
    return get_scores(correlation_scores)


@pytest.mark.parametrize(
    "test_attribute",
    [
        "dark",
        "blurry",
        "odd_aspect_ratio",
    ],
)
def test_correlation_scores_against_standard(test_attribute):
    """
    Tests that correlation scores for specific filters are lower than standard scores.

    Asserts:
        AssertionError: If any of the specific filter scores are not lower than the standard scores.
    """
    standard_correlation_scores = get_correlation_scores()
    attribute_filter_scores = get_correlation_scores(circle_filter=f"{test_attribute}")
    assert (
        standard_correlation_scores[f"{test_attribute}_score"]
        > attribute_filter_scores[f"{test_attribute}_score"]
    )


@pytest.mark.parametrize(
    "test_attribute",
    [
        "dark",
        pytest.param(
            "blurry",
            marks=pytest.mark.xfail(
                reason="odd aspect ratio filter seems to score lower", strict=True
            ),
        ),
        "odd_aspect_ratio",
    ],
)
def test_smallest_scores_with_filters(test_attribute):
    """
    Tests that each specific filter has the smallest correlation score for its respective property.

    Asserts:
        AssertionError: If any specific filter score is not the smallest for its respective property.
    """

    attributes_to_score = ["dark", "blurry", "odd_aspect_ratio"]
    standard_correlation_scores = get_correlation_scores()

    score_key = f"{test_attribute}_score"
    filtered_scores = {f: get_correlation_scores(circle_filter=f) for f in attributes_to_score}

    # The attribute being tested should have the lowest score for the filtered dataset
    test_scores = filtered_scores.pop(test_attribute)
    assert test_scores[score_key] <= min(
        standard_correlation_scores[score_key],
        *[scores[score_key] for scores in filtered_scores.values()],
    )
