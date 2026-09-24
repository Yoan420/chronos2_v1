"""Rebuild the code-drawn Windows icon; Pillow is only a build-time dependency."""
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    scale = 4
    image = Image.new('RGBA', (256 * scale, 256 * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    def points(values):
        return [(x * scale, y * scale) for x, y in values]
    draw.rounded_rectangle((8*scale, 8*scale, 248*scale, 248*scale), radius=46*scale, fill='#101216', outline='#b78149', width=3*scale)
    draw.line(points([(49,66),(32,66),(32,49),(49,49)]), fill='#759b9d', width=3*scale)
    draw.line(points([(207,190),(224,190),(224,207),(207,207)]), fill='#759b9d', width=3*scale)
    pulse = points([(35,143),(67,143),(90,82),(123,198),(159,56),(185,143),(221,143)])
    for width, color in [(19,'#362922'),(12,'#6b4a2e'),(6,'#ffba68')]:
        draw.line(pulse, fill=color, width=width*scale, joint='curve')
    image = image.resize((256,256), Image.Resampling.LANCZOS)
    image.save(Path(__file__).parent / 'static' / 'chronos.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])


if __name__ == '__main__':
    main()
