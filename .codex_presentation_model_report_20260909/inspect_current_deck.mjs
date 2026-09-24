import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
console.log(`slides=${presentation.slides.count}`);
for (let index = 0; index < presentation.slides.count; index += 1) {
  const slide = presentation.slides.getItem(index);
  console.log(`slide ${index + 1} id=${slide.id} charts=${slide.charts.items.length}`);
  slide.shapes.items.forEach((shape, shapeIndex) => {
    const text = shape.text ? String(shape.text).replace(/\s+/g, " ").trim() : "";
    if (text) console.log(`  shape ${shapeIndex} ${shape.placeholderType ?? ""}: ${text.slice(0, 180)}`);
  });
}
