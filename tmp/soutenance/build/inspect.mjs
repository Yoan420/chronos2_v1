import fs from 'node:fs/promises';
import {PresentationFile, FileBlob} from '@oai/artifact-tool';
const root='C:/Users/BQ6757/chronos2_v1/tmp/soutenance';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/BQ6757/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx'));
await fs.mkdir(root+'/template',{recursive:true});
await fs.writeFile(root+'/template/inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,image,table,chart,layout',maxChars:300000})).ndjson);
await fs.writeFile(root+'/template/proto.json',JSON.stringify(p.toProto(),null,2));
for (let i=0;i<p.slides.items.length;i++) {
 const s=p.slides.items[i];
 const b=await p.export({slide:s,format:'png',scale:0.8});
 await fs.writeFile(`${root}/template/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
 await fs.writeFile(`${root}/template/slide-${i+1}.json`,await (await s.export({format:'layout'})).text());
 console.log('Rendered '+(i+1));
}
