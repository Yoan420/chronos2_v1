import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const presentation = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
console.log(JSON.stringify(presentation.theme?.colorScheme ?? presentation.theme, null, 2));
