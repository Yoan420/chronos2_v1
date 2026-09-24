import fs from 'node:fs/promises';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/BQ6757/chronos2_v1';
const html=await fs.readFile(root+'/runs/exports/2026-09-09/fr/nuclear_kalman/forecast_fr_2026-09-09_nuclear_kalman.html','utf8');
function takeJSON(s,start){
 let depth=0,inString=false,escape=false;
 for(let i=start;i<s.length;i++){
  const c=s[i];
  if(inString){if(escape){escape=false;}else if(c==='\\'){escape=true;}else if(c==='"'){inString=false;}}
  else if(c==='"'){inString=true;}
  else if(c==='['||c==='{'){depth++;}
  else if(c===']'||c==='}') {depth--;if(depth===0)return {value:JSON.parse(s.slice(start,i+1)),end:i+1};}
 }
 throw new Error('Unterminated JSON');
}
const plots=[];
for(const match of html.matchAll(/Plotly\.newPlot\(\s*"([^"]+)",\s*/g)){
 const d=takeJSON(html,match.index+match[0].length);
 const layoutStart=html.indexOf('{',d.end);
 const l=takeJSON(html,layoutStart);
 plots.push({id:match[1],data:d.value,layout:l.value});
}
const payloadMatch=/const payload\s*=\s*/.exec(html);
const payload=takeJSON(html,payloadMatch.index+payloadMatch[0].length).value;
const out=root+'/.codex_presentation_report_additions_20260909/report_data.json';
await fs.writeFile(out,JSON.stringify({plots,payload}));
console.log('Plots',plots.map(p=>({id:p.id,title:p.layout.title,series:p.data.map(t=>({name:t.name,type:t.type,count:t.x?.length,keys:Object.keys(t)}))})));
console.log('Payload keys',Object.keys(payload),'record sample',JSON.stringify(payload.records?.slice(0,2)));
const pres=await PresentationFile.importPptx(await FileBlob.load(root+'/deliverables/Presentation_Chronos2_FR_2026-08-26.pptx'));
console.log((await pres.inspect({kind:'layout',maxChars:4000})).ndjson);
console.log('Masters',pres.masters.items.map(m=>({id:m.id,name:m.name})), 'Layouts',pres.layouts.items.map(l=>({id:l.id,summary:l.placeholders.summary()})));
for(let i=0;i<pres.slides.count;i++)console.log(JSON.stringify({slide:i+1,shapes:pres.slides.getItem(i).shapes.items.map(s=>({id:s.id,name:s.name,position:s.position,text:String(s.text??''),style:s.text?.style}))}));
