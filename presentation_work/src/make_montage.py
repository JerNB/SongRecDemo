from pathlib import Path
from math import ceil
from PIL import Image, ImageDraw

src = Path(r"C:\SeniorProj\presentation_work\scratch\pptx-previews")
files = sorted(src.glob("*.png"))

thumb_w, thumb_h = 480, 270
margin = 24
label_h = 28
cols = 2
rows = ceil(len(files) / cols)

canvas = Image.new(
    "RGB",
    (cols * thumb_w + (cols + 1) * margin, rows * (thumb_h + label_h) + (rows + 1) * margin),
    "white",
)
draw = ImageDraw.Draw(canvas)

for idx, file_path in enumerate(files):
    image = Image.open(file_path).convert("RGB")
    image.thumbnail((thumb_w, thumb_h))
    x = margin + (idx % cols) * (thumb_w + margin)
    y = margin + (idx // cols) * (thumb_h + label_h + margin)
    canvas.paste(image, (x, y))
    draw.text((x, y + thumb_h + 6), file_path.stem, fill=(24, 33, 43))

out = Path(r"C:\SeniorProj\presentation_work\scratch\pptx-preview-montage.png")
canvas.save(out)
print(out)
