import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
const shape = presentation.slides.getItem(0).shapes.getItemAt(0);
let current = shape;
for (let depth = 0; depth < 8 && current; depth += 1) {
  console.log(`depth ${depth}`, Object.getOwnPropertyNames(current));
  current = Object.getPrototypeOf(current);
}
console.log("text", shape.text);
console.log("text own", Object.getOwnPropertyNames(shape.text));
console.log("text proto", Object.getOwnPropertyNames(Object.getPrototypeOf(shape.text)));
console.log("text string", String(shape.text));
console.log("paragraph count", shape.paragraphs?.length, shape.getParagraphs?.().length);
console.log("paragraph proto", shape.getParagraphs?.()[0] ? Object.getOwnPropertyNames(Object.getPrototypeOf(shape.getParagraphs()[0])) : null);
console.log("paragraphs", shape.getParagraphs?.().map((p) => ({ text: p.text, runs: p.runs?.items?.map((r) => r.text) })));
console.log("textFrame", shape.textFrame ? Object.getOwnPropertyNames(Object.getPrototypeOf(shape.textFrame)) : null);
console.log("textRange", shape.textFrame?.textRange ? Object.getOwnPropertyNames(Object.getPrototypeOf(shape.textFrame.textRange)) : null);
console.log("tf text", shape.textFrame?.text);
console.log("tr text", shape.textFrame?.textRange?.text);
