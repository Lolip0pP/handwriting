from docx import Document
from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import random

default_config = {
    "page_size": (1240, 1754),  # A4 @ 150 DPI
    "page_color": (255, 255, 255),
    "text_color": (0, 0, 0),
    "font_size": 40,
    "margin_left": 100,
    "margin_top": 70,
    "margin_right": 30,
    "margin_bottom": 50,
    "line_gap": 15,
    "word_space": 0.6,
    "font_name": "fonts/Katherine-Plus.ttf",  # <-- Replace with your TTF/OTF
    "seed": 17,
}


def create_page(config: dict) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    page_size = config["page_size"]
    img_arr = np.zeros((page_size[1], page_size[0], 3), dtype=np.uint8)
    img_arr[:, :] = config["page_color"]
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    return img, draw


def render_page(args):
    text_chunk, config = args
    random.seed(config["seed"])  # Set seed for reproducible randomness
    img, draw = create_page(config)

    x, y = config["margin_left"], config["margin_top"]
    page_width, page_height = config["page_size"]

    for line in text_chunk.split("\n"):
        words = line.split()

        for word in words:
            # Randomize font size (±10% of base font size)
            font_size = int(config["font_size"] * random.uniform(0.9, 1.1))
            font = ImageFont.truetype(config["font_name"], font_size)

            # Calculate word width and height
            word_width = draw.textlength(word, font=font)
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_height = word_bbox[3] - word_bbox[1]

            if x + word_width + config["margin_right"] >= page_width:
                # Move to next line
                x = config["margin_left"]
                y += config["font_size"] + config["line_gap"]

            # Check if y is within page bounds
            if y + word_height + config["margin_bottom"] >= page_height:
                print(
                    f"Warning: Skipping word '{word}' as it exceeds page height."
                )  # Debug
                continue

            # Random rotation (-5 to 5 degrees) and slight position offset
            rotation = random.uniform(-3, 3)
            x_offset = random.uniform(-3, 3)
            y_offset = random.uniform(-2, 2)

            # rotation = 0
            # x_offset = 0
            # y_offset = 0

            if rotation == 0 and x_offset == 0 and y_offset == 0:
                # Render directly without rotation or offsets
                draw.text((x, y), word, config["text_color"], font=font)
                x += word_width + int(config["word_space"] * config["font_size"])
            else:
                # Create a temporary image for the rotated text (left-aligned)
                extra = int(max(word_width, word_height) * 0.2) + 40
                temp_size = (int(word_width + extra), int(word_height + extra))
                text_img = Image.new("RGBA", temp_size, (0, 0, 0, 0))
                text_draw = ImageDraw.Draw(text_img)
                text_draw.text(
                    (extra // 2, extra // 2), word, config["text_color"], font=font
                )
                text_img = text_img.rotate(rotation, expand=True)

                # Get the bounding box and crop to tight bounds
                rotated_bbox = text_img.getbbox()
                if rotated_bbox:
                    text_img = text_img.crop(rotated_bbox)
                    rotated_width = rotated_bbox[2] - rotated_bbox[0]
                    rotated_height = rotated_bbox[3] - rotated_bbox[1]
                else:
                    rotated_width = word_width
                    rotated_height = word_height

                # Calculate paste position (approximate left-baseline alignment) and clamp
                paste_x = max(
                    config["margin_left"],
                    min(
                        int(x + x_offset),
                        page_width - rotated_width - config["margin_right"],
                    ),
                )
                paste_y = max(
                    config["margin_top"],
                    min(
                        int(y + y_offset + word_bbox[1]),
                        page_height - rotated_height - config["margin_bottom"],
                    ),
                )

                # Paste the rotated text
                img.paste(text_img, (paste_x, paste_y), text_img)

                # Update x with original width to maintain spacing (approx for small rotations)
                x += word_width + int(config["word_space"] * config["font_size"])

        y += config["font_size"] + config["line_gap"]
        x = config["margin_left"]

    return img


def simulate_chunks(text: str, config: dict) -> list[str]:
    paragraphs = text.split("\n")
    chunks = []
    current_chunk_lines = []
    y = config["margin_top"]
    font_size = int(config["font_size"] * 1.1)  # Conservative for simulation
    line_height = font_size + config["line_gap"]
    max_y = config["page_size"][1] - config["margin_bottom"]
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    font = ImageFont.truetype(config["font_name"], font_size)

    for paragraph in paragraphs:
        words = paragraph.split()
        current_line_words = []
        x = config["margin_left"]
        for word in words:
            word_width = draw.textlength(word, font=font)
            if x + word_width + config["margin_right"] >= config["page_size"][0]:
                if current_line_words:
                    current_chunk_lines.append(" ".join(current_line_words))
                current_line_words = []
                x = config["margin_left"]
                y += line_height
                if y + line_height > max_y:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
                    y = config["margin_top"]
            current_line_words.append(word)
            x += word_width + int(config["word_space"] * font_size)
        if current_line_words:
            current_chunk_lines.append(" ".join(current_line_words))
        y += line_height
        x = config["margin_left"]
        if y + line_height > max_y:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            y = config["margin_top"]
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
    print(f"Simulated chunks: {len(chunks)}")  # Debug
    return chunks


def docx_to_pdf(docx_path: str, output_pdf: str, config: dict = None):
    if config is None:
        config = default_config

    # Read docx
    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    print(f"Total characters in input: {len(text)}")  # Debug

    # Simulate chunks for proper page breaks
    chunks = simulate_chunks(text, config)

    # Parallel rendering
    args = [(chunk, config) for chunk in chunks]
    with Pool(processes=cpu_count()) as pool:
        pages = list(tqdm(pool.imap(render_page, args), total=len(args)))

    # Save PDF
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:], format="PDF")
    print(f"PDF saved at {output_pdf}")


if __name__ == "__main__":
    docx_to_pdf("input.docx", "output.pdf", config=default_config)
