from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import random
from pathlib import Path
import math


WIDTH = 1200
HEIGHT = 800

FONT_SIZE = 40

LEFT_MARGIN = 100
RIGHT_MARGIN = 100
TOP_MARGIN = 90
BOTTOM_MARGIN = 90

LINE_SPACING = 18

IMAGES_PER_SCRIPT = 100


TRAIN_COUNT = 85
VAL_COUNT = 10
TEST_COUNT = 5

ENABLED_SCRIPTS = ["devanagari", "modi", "sharada"]


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FONTS_DIR = BASE_DIR / "fonts"
OUTPUT_DIR = BASE_DIR / "output"



SCRIPT_CONFIG = {
    "devanagari": {
        "corpus": DATA_DIR / "devanagari_md.md",
        "font": FONTS_DIR / "NotoSansDevanagari-Regular.ttf"
    },
    "modi": {
        "corpus": DATA_DIR / "Modi_md.md",
        "font": FONTS_DIR / "NotoSansModi-Regular.ttf"
    },
    "sharada": {
        "corpus": DATA_DIR / "sharada_md.md",
        "font": FONTS_DIR / "Sharada.ttf"
    }
}


random.seed()




def load_corpus(corpus_path):

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}"
        )

    with open(
        corpus_path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not lines:
        raise ValueError(
            f"Corpus is empty: {corpus_path}"
        )

    return lines



def create_background():

  

    base = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.float32
    )

    # Warm aged paper
    base[:, :] = [210, 190, 150]


    

    fine_noise = np.random.normal(
        0,
        5,
        (HEIGHT, WIDTH, 1)
    )


    

    small_h = 80
    small_w = 120

    large_noise = np.random.normal(
        0,
        7,
        (small_h, small_w)
    )

    large_noise = Image.fromarray(
        large_noise.astype(np.float32),
        mode="F"
    )

    large_noise = large_noise.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.BICUBIC
    )

    large_noise = np.array(
        large_noise
    )[:, :, np.newaxis]



    aging = np.random.normal(
        0,
        10,
        (40, 60)
    )

    aging = Image.fromarray(
        aging.astype(np.float32),
        mode="F"
    )

    aging = aging.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.BICUBIC
    )

    aging = np.array(
        aging
    )[:, :, np.newaxis]

    # Aging mostly darkens
    aging = np.minimum(
        aging,
        0
    )


    texture = (
        base
        + fine_noise
        + large_noise
        + aging
    )


    stains = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32
    )

    number_of_stains = random.randint(
        12,
        22
    )

    yy, xx = np.ogrid[
        :HEIGHT,
        :WIDTH
    ]

    for _ in range(number_of_stains):

        x = np.random.randint(
            0,
            WIDTH
        )

        y = np.random.randint(
            0,
            HEIGHT
        )

        radius = np.random.randint(
            20,
            90
        )

        distance = np.sqrt(
            (xx - x) ** 2
            +
            (yy - y) ** 2
        )

        stain = np.exp(
            -(distance ** 2)
            /
            (2 * radius ** 2)
        )

        strength = np.random.uniform(
            5,
            20
        )

        stains += (
            stain * strength
        )


    stains = np.clip(
        stains,
        0,
        30
    )

    texture -= (
        stains[:, :, np.newaxis]
    )


    # --------------------------------------------------------
    # EDGE DARKENING
    # --------------------------------------------------------

    center_x = WIDTH / 2
    center_y = HEIGHT / 2

    distance_from_center = np.sqrt(
        ((xx - center_x) / center_x) ** 2
        +
        ((yy - center_y) / center_y) ** 2
    )

    edge_strength = np.clip(
        distance_from_center,
        0,
        1
    )

    edge_strength = edge_strength ** 2

    edge_darkening = (
        edge_strength * 18
    )

    texture -= (
        edge_darkening[:, :, np.newaxis]
    )


    if random.random() < 0.45:

        fold_x = random.randint(
            150,
            WIDTH - 150
        )

        fold_width = random.randint(
            2,
            6
        )

        fold_strength = random.uniform(
            5,
            15
        )

        fold = np.exp(
            -(
                (np.arange(WIDTH) - fold_x) ** 2
            )
            /
            (2 * fold_width ** 2)
        )

        fold = np.tile(
            fold,
            (HEIGHT, 1)
        )

        texture -= (
            fold[:, :, np.newaxis]
            * fold_strength
        )


    texture = np.clip(
        texture,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        texture,
        "RGB"
    )



def select_text_block(lines):

    start_index = random.randint(
        0,
        max(0, len(lines) - 1)
    )

    number_of_lines = random.randint(
        6,
        14
    )

    selected_lines = lines[
        start_index:
        start_index + number_of_lines
    ]

    return " ".join(
        selected_lines
    )


def wrap_text(
    draw,
    text,
    font,
    max_width
):

    words = text.split()

    output_lines = []

    current_line = ""

    for word in words:

        if current_line:

            test_line = (
                current_line
                + " "
                + word
            )

        else:

            test_line = word


        bbox = draw.textbbox(
            (0, 0),
            test_line,
            font=font
        )

        test_width = (
            bbox[2]
            -
            bbox[0]
        )


        if test_width <= max_width:

            current_line = test_line

        else:

            if current_line:

                output_lines.append(
                    current_line
                )

            current_line = word


    if current_line:

        output_lines.append(
            current_line
        )

    return output_lines


def render_text(
    image,
    font,
    text
):

    draw = ImageDraw.Draw(
        image
    )

    max_width = (
        WIDTH
        -
        LEFT_MARGIN
        -
        RIGHT_MARGIN
    )



    lines_to_draw = wrap_text(
        draw,
        text,
        font,
        max_width
    )


    x = LEFT_MARGIN

    y = TOP_MARGIN



    annotations = []



    for line_index, line in enumerate(
        lines_to_draw
    ):

        bbox = draw.textbbox(
            (x, y),
            line,
            font=font
        )

        line_width = (
            bbox[2]
            -
            bbox[0]
        )

        line_height = (
            bbox[3]
            -
            bbox[1]
        )



        if (
            y + line_height
            >
            HEIGHT - BOTTOM_MARGIN
        ):
            break



        x_jitter = random.randint(
            -5,
            5
        )

        y_jitter = random.randint(
            -2,
            2
        )



        ink_variation = random.randint(
            -10,
            8
        )

        ink_color = (
            max(20, 50 + ink_variation),
            max(15, 35 + ink_variation),
            max(10, 20 + ink_variation)
        )



        padding = 30

        temp_width = (
            int(line_width)
            +
            padding * 2
        )

        temp_height = (
            int(line_height)
            +
            padding * 2
        )

        line_layer = Image.new(
            "RGBA",
            (
                temp_width,
                temp_height
            ),
            (0, 0, 0, 0)
        )

        line_draw = ImageDraw.Draw(
            line_layer
        )


        
        line_draw.text(
            (
                padding,
                padding
            ),
            line,
            font=font,
            fill=(
                ink_color[0],
                ink_color[1],
                ink_color[2],
                random.randint(
                    210,
                    245
                )
            )
        )



        rotation = random.uniform(
            -1.2,
            1.2
        )

        line_layer = line_layer.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True
        )



        if random.random() < 0.20:

            line_layer = line_layer.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(
                        0.15,
                        0.45
                    )
                )
            )



        paste_x = (
            x
            + x_jitter
            - padding
        )

        paste_y = (
            y
            + y_jitter
            - padding
        )


        
        paste_x = max(
            LEFT_MARGIN,
            paste_x
        )

       
        image.paste(
            line_layer,
            (
                int(paste_x),
                int(paste_y)
            ),
            line_layer
        )



        annotation_bbox = [
            int(paste_x),
            int(paste_y),
            int(
                paste_x
                + line_layer.width
            ),
            int(
                paste_y
                + line_layer.height
            )
        ]


        annotations.append({
            "text": line,
            "bbox": annotation_bbox
        })



        y += (
            line_height
            +
            LINE_SPACING
            +
            random.randint(
                -2,
                3
            )
        )


    return annotations



def add_ink_imperfections(
    image,
    annotations
):

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(
        overlay
    )


    for annotation in annotations:

        x1, y1, x2, y2 = (
            annotation["bbox"]
        )


        
        if random.random() < 0.35:

            for _ in range(
                random.randint(1, 3)
            ):

                x = random.randint(
                    max(0, x1),
                    min(WIDTH - 1, x2)
                )

                y = random.randint(
                    max(0, y1),
                    min(HEIGHT - 1, y2)
                )

                radius = random.randint(
                    2,
                    5
                )

                draw.ellipse(
                    (
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius
                    ),
                    fill=(
                        55,
                        40,
                        25,
                        random.randint(
                            20,
                            60
                        )
                    )
                )


    overlay = overlay.filter(
        ImageFilter.GaussianBlur(
            radius=1.2
        )
    )


    image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay
    )

    return image.convert(
        "RGB"
    )



def create_annotation(
    script,
    image_name,
    source_text,
    annotations
):

    lines = []

    lines.append(
        "# Synthetic Manuscript Annotation"
    )

    lines.append("")

    lines.append(
        f"script: {script}"
    )

    lines.append(
        f"image: {image_name}"
    )

    lines.append(
        f"image_width: {WIDTH}"
    )

    lines.append(
        f"image_height: {HEIGHT}"
    )

    lines.append("")

    lines.append(
        "## Source Text"
    )

    lines.append("")

    lines.append(
        source_text
    )

    lines.append("")

    lines.append(
        "## Text Regions"
    )

    lines.append("")

    for index, item in enumerate(
        annotations,
        start=1
    ):

        bbox = item["bbox"]

        lines.append(
            f"### Region {index}"
        )

        lines.append(
            f"text: {item['text']}"
        )

        lines.append(
            "bbox: "
            f"[{bbox[0]}, "
            f"{bbox[1]}, "
            f"{bbox[2]}, "
            f"{bbox[3]}]"
        )

        lines.append("")


    return "\n".join(
        lines
    )



def generate_one(
    script,
    index,
    split,
    corpus_lines,
    font
):

   
    image = create_background()


   
    source_text = select_text_block(
        corpus_lines
    )


    annotations = render_text(
        image,
        font,
        source_text
    )


    
    image = add_ink_imperfections(
        image,
        annotations
    )


   

    image_dir = (
        OUTPUT_DIR
        / script
        / split
        / "images"
    )

    annotation_dir = (
        OUTPUT_DIR
        / script
        / split
        / "annotations"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    image_name = (
        f"{script}_{index:04d}.png"
    )

    annotation_name = (
        f"{script}_{index:04d}.md"
    )



    image_path = (
        image_dir
        / image_name
    )

    image.save(
        image_path
    )



    annotation_text = create_annotation(
        script,
        image_name,
        source_text,
        annotations
    )

    annotation_path = (
        annotation_dir
        / annotation_name
    )

    annotation_path.write_text(
        annotation_text,
        encoding="utf-8"
    )



def generate_dataset():

    for script in ENABLED_SCRIPTS:

        print()
        print(
            "=" * 60
        )

        print(
            f"Generating: {script}"
        )

        print(
            "=" * 60
        )


        config = SCRIPT_CONFIG[
            script
        ]



        if not config["font"].exists():

            print(
                f"WARNING: Font missing for {script}"
            )

            print(
                f"Expected: {config['font']}"
            )

            print(
                "Skipping this script."
            )

            continue


  

        corpus_lines = load_corpus(
            config["corpus"]
        )



        font = ImageFont.truetype(
            str(config["font"]),
            FONT_SIZE
        )


       

        for i in range(
            1,
            TRAIN_COUNT + 1
        ):

            generate_one(
                script,
                i,
                "train",
                corpus_lines,
                font
            )

            print(
                f"\rTrain: {i}/{TRAIN_COUNT}",
                end=""
            )


        print()



        for i in range(
            1,
            VAL_COUNT + 1
        ):

            generate_one(
                script,
                i,
                "validation",
                corpus_lines,
                font
            )

            print(
                f"\rValidation: {i}/{VAL_COUNT}",
                end=""
            )


        print()



        for i in range(
            1,
            TEST_COUNT + 1
        ):

            generate_one(
                script,
                i,
                "test",
                corpus_lines,
                font
            )

            print(
                f"\rTest: {i}/{TEST_COUNT}",
                end=""
            )


        print()

        print(
            f"{script} completed."
        )




if __name__ == "__main__":

    print(
        "Synthetic Manuscript Generator"
    )

    print(
        f"Image size: {WIDTH}x{HEIGHT}"
    )

    print(
        f"Images per script: {IMAGES_PER_SCRIPT}"
    )

    generate_dataset()

    print()
    print(
        "Dataset generation completed!"
    )