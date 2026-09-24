import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const workspaceDir='C:/Users/BQ6757/chronos2_v1';
const root=workspaceDir+'/tmp/soutenance';
const SKILL_DIR='C:/Users/BQ6757/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const RUNTIME_PYTHON='C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const ref='C:/Users/BQ6757/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const out=workspaceDir+'/output/soutenance';
const family='Helvetica Neue';
await fs.mkdir(out,{recursive:true});
await fs.mkdir(root+'/draft',{recursive:true});
const imported=await PresentationFile.importPptx(await FileBlob.load(ref));
const proto=imported.toProto();
const sourceMap=[1,5,5,5,17,14,13,11,13,26,13,14];
proto.slides=sourceMap.map((num,i)=>{
  const s=structuredClone(proto.slides[num-1]);
  s.id='soutenance_'+(i+1); s.index=i;
  // Select content placeholders while preserving masters, layouts and slide chrome.
  if ([3,4].includes(i+1)) s.elements=s.elements.filter(e=>['3','532','533'].includes(e.id));
  if ([6,12].includes(i+1)) s.elements=s.elements.filter(e=>e.shape?.placeholder?.type || false);
  return s;
});
// Source table slides can contain a different placeholder encoding. Retain their
// title, slide number and footer by the original inspected identifiers.
for (const ix of [5,11]) {
 const source=imported.toProto().slides[13];
 const layout=JSON.parse(await fs.readFile(root+'/template/slide-14.json','utf8'));
 const chromeIds=layout.elements.filter(e=>/Footer|Slide Number|Google Shape;532|Title 2|Google Shape;533/.test(e.name||'')).map(e=>e.id);
 proto.slides[ix].elements=structuredClone(source.elements.filter(e=>chromeIds.includes(e.id)));
}
const p=Presentation.load(proto);
const slides=p.slides.items;
const shapeById=(s,id)=>s.shapes.items.find(x=>x.id===id);
const sourceLayouts=await Promise.all(sourceMap.map(n=>fs.readFile(root+'/template/slide-'+n+'.json','utf8').then(JSON.parse)));
function setText(sh,text,size=24,extra={}) {
 if(!sh)throw new Error('Missing shape for '+text);
 const paras=text.split('\n').map(line=>({runs:[{run:line,textStyle:{fontSize:size+'px',typeface:family}}],bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:0}));
 sh.text=paras;
 sh.text.style={fontSize:size,typeface:family,color:'#000000',alignment:'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{top:0,bottom:0,left:0,right:0},...extra};
 return sh;
}
function addText(s,text,box,size=24,extra={}) {
 const sh=s.shapes.add({geometry:'textbox',name:text.slice(0,50),position:box,fill:'none',line:{fill:'none',width:0}});
 return setText(sh,text,size,extra);
}
function at(s,text,x,y,w,h,size=24,extra={}) {return addText(s,text,{left:x,top:y,width:w,height:h},size,extra);}
function findEntry(slideNo,regex){return sourceLayouts[slideNo-1].elements.find(e=>regex.test(e.name||''));}
function title(s,n,text){
 const l=sourceLayouts[n-1];
 const e=l.elements.find(e=>/title/i.test(e.name||'')&&!/subtitle/i.test(e.name||'')) || l.elements.find(e=>/Google Shape;533/.test(e.name||''));
 return setText(shapeById(s,e.id),text,38.67);
}
function footer(s,n,label){
 const l=sourceLayouts[n-1];
 const f=l.elements.find(e=>/Footer/.test(e.name||''));
 if(f&&shapeById(s,f.id)) setText(shapeById(s,f.id),label,13.33,{verticalAlignment:'bottom'});
 const e=l.elements.find(e=>/Slide Number|Google Shape;532/.test(e.name||''));
 if(e&&shapeById(s,e.id)) setText(shapeById(s,e.id),String(n),13.33,{alignment:'right',verticalAlignment:'bottom'});
}
function block(sh,head,body,size=24){
 setText(sh,head+'\n\n'+body,size);
 sh.text.get(head).fontSize=32;
 return sh;
}
function columns(n){const s=slides[n-1];return sourceLayouts[n-1].elements.filter(e=>e.text?.includes('Lorem')&&!/Subtitle/.test(e.name||'')).map(e=>shapeById(s,e.id)).filter(Boolean);}
function grid(n,items){
 const s=slides[n-1];
 const arr=sourceLayouts[n-1].elements.filter(e=>e.text?.startsWith('Title goes here')).sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]);
 items.forEach((item,i)=>block(shapeById(s,arr[i].id),...item));
}
async function img(s,name,box){s.images.add({blob:new Uint8Array(await fs.readFile(root+'/'+name)),contentType:'image/png',position:box,fit:'contain',alt:name==='figure-1.png'?'Schéma du mémoire : calcul d’une tête d’auto-attention':'Schéma du mémoire : tokens temporels PatchTST et tokens de variables iTransformer'});}
function table(s,values,top,widths,height=380,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:41.33,top,width:1197.33,height,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};
 t.borders.assign({fill:'#FFFFFF',width:1,style:'solid'});
 t.rows[0].height=58;
 for(let r=1;r<values.length;r++)t.rows[r].height=(height-58)/(values.length-1);
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#EAEAEA':(r%2?'#F6F6F6':'#FFFFFF');
  cell.text.style={typeface:family,fontSize:size,color:'#000000',bold:r===0,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:{top:12,bottom:12,left:14,right:14}};
 }
 return t;
}

// 1. Cover keeps the selected template's title composition.
{
const s=slides[0];
setText(shapeById(s,'4'),'Transformers et modèles\nde fondation pour les\nséries temporelles',72,{verticalAlignment:'bottom',lineSpacing:0.95});
setText(shapeById(s,'5'),'Yoan Kesraoui\nSorbonne Data Analytics\nMémoire de fin d’études 2025-2026',24);
setText(shapeById(s,'6'),'Soutenance de mémoire',24);
}
// 2. Research question and scope.
{
const n=2,s=slides[n-1];title(s,n,'Problématique et démarche');footer(s,n,'Revue de littérature');
const [a,b]=columns(n);
setText(a,'Comment les Transformers\napprennent-ils à prévoir,\net à quelles conditions\ntransfèrent-ils leurs acquis\nà de nouvelles séries ?',39);
block(b,'Une revue critique','Articles d’origine et travaux consultés\njusqu’au 9 septembre 2026\n\nGrille de lecture commune :\nentrées, interactions, apprentissage, sorties\n\nAucune nouvelle campagne expérimentale',24);
}
// 3. Original mechanism figure from the thesis.
{
const n=3,s=slides[n-1];title(s,n,'L’attention combine l’information accessible');footer(s,n,'Mémoire, figure 1, p. 7');
await img(s,'figure-1.png',{left:80,top:175,width:1120,height:430});
at(s,'Positions : ordre temporel. Masquage : accès aux informations admissibles.',41.33,605,1197.33,40,25);
}
// 4. Original representation comparison.
{
const n=4,s=slides[n-1];title(s,n,'Le token détermine ce que l’attention compare');footer(s,n,'Mémoire, figure 3, p. 14');
await img(s,'figure-3.png',{left:80,top:150,width:1120,height:485});
}
// 5. Retained text timeline.
{
const n=5,s=slides[n-1];title(s,n,'Le préentraînement permet de réutiliser le modèle');footer(s,n,'Zero-shot et adaptation sont deux régimes distincts');
const l=sourceLayouts[n-1];
const labels=l.elements.filter(e=>e.text==='Date').sort((a,b)=>a.bbox[0]-b.bbox[0]);
const bodies=l.elements.filter(e=>e.text?.startsWith('Title here')).sort((a,b)=>a.bbox[0]-b.bbox[0]);
['Préentraînement','Nouvelle série','Prévision'].forEach((t,i)=>{const sh=shapeById(s,labels[i].id);sh.position={left:labels[i].bbox[0],top:labels[i].bbox[1],width:350,height:36};setText(sh,t,26);});
[['Des séries variées','Domaines et fréquences variés\nApprentissage des paramètres'],['Un nouvel historique','Le contexte change\nLes paramètres restent fixes'],['Un futur à estimer','Réutilisation du modèle\nSans mise à jour par gradient']].forEach((v,i)=>block(shapeById(s,bodies[i].id),...v,24));
const line=l.elements.find(e=>e.geometry==='straightConnector1');shapeById(s,line.id).position={left:41.33,top:354.2,width:1197.33,height:0.03};
at(s,'Zero-shot : paramètres fixes sur la tâche cible',41.33,177,1197.33,58,36);
at(s,'Le fine-tuning modifie tout ou partie des paramètres.',41.33,600,1197.33,38,24);
}
// 6. Editable comparison. Version labels avoid mixing generations.
{
const n=6,s=slides[n-1];title(s,n,'Trois méthodes, des sorties différentes');footer(s,n,'Versions décrites dans le mémoire, p. 18-20');
at(s,'La représentation et la sortie font partie du choix du modèle.',41.33,150,1197.33,60,30);
table(s,[['Modèle étudié','Représentation et interactions','Sortie'],['Chronos initial\n(2024)','Valeurs quantifiées\nGénération séquentielle','Trajectoires échantillonnées'],['TimesFM initial\n(2024)','Patches continus\nAttention temporelle causale','Blocs de valeurs ponctuelles'],['Chronos-2\n(2025)','Patches continus\nAttention temporelle et entre séries','Quantiles directs']],230,[255,557,385],370,24);
}
// 7. Four fair-comparison dimensions.
{
const n=7,s=slides[n-1];title(s,n,'Un bon score ne suffit pas à établir le transfert');footer(s,n,'Mémoire, p. 24-25');
grid(n,[['Nouveauté des données','Le zero-shot ne garantit pas\nl’absence d’exposition au préentraînement.'],['Informations comparables','Mêmes séries, dates et horizons.\nCovariables connues au moment de prévoir.'],['Qualité et coût','Précision, durée et mémoire.\nUne référence simple sur les mêmes tâches.'],['Changement de régime','Une baisse sur de nouvelles données\nne prouve pas une contamination passée.']]);
}
// 8. Mathematical example from thesis, not empirical performance.
{
const n=8,s=slides[n-1];title(s,n,'Les quantiles ne décrivent pas toute la dépendance');footer(s,n,'Exemple théorique du mémoire, p. 25-26');
const l=sourceLayouts[n-1];
const intro=l.elements.find(e=>e.text?.startsWith('Topic'));
setText(shapeById(s,intro.id),'Deux événements ont chacun 50 % de probabilité.\nQuelle est la probabilité qu’ils surviennent ensemble ?',32);
const stats=l.elements.filter(e=>e.text?.startsWith('Lorem')).sort((a,b)=>a.bbox[0]-b.bbox[0]);
['25 %','50 %'].forEach((t,i)=>setText(shapeById(s,stats[i].id),t,72));
const detail=l.elements.filter(e=>e.text?.startsWith('Detail')).sort((a,b)=>a.bbox[0]-b.bbox[0]);
setText(shapeById(s,detail[0].id),'Événements indépendants\n0,5 × 0,5 = 0,25',28);
setText(shapeById(s,detail[1].id),'Événements toujours simultanés\nProbabilité de réalisation commune : 0,5',28);
at(s,'Même probabilité à chaque horizon, scénarios différents.',41.33,612,1197.33,35,26);
}
// 9. Proposed experiments, with visible status.
{
const n=9,s=slides[n-1];title(s,n,'Quatre protocoles pour tester le transfert');footer(s,n,'Critères et budgets à fixer avant le test');
at(s,'Expériences proposées, à réaliser',41.33,147,1197.33,42,28);
grid(n,[['H1  Transfert temporel','Geler le modèle, enregistrer les prévisions,\npuis évaluer les observations futures.'],['H2  Covariables','Comparer la cible seule aux variables\ncorrectes, retardées ou perturbées.'],['H3  Adaptation','Comparer modèle gelé, LoRA, correction\nrésiduelle et ajustement complet.\nMesurer le coût et l’oubli.'],['H4  Trajectoires','Comparer les dépendances à marges identiques.\nÉvaluer les événements sur plusieurs horizons.']]);
}
// 10. Main conclusion.
{
const s=slides[9];setText(shapeById(s,'4'),'Le transfert dépend\ndes conditions d’usage',80,{verticalAlignment:'bottom'});
const sub=shapeById(s,'5');sub.position={left:41.33,top:522.13,width:1160,height:114};
setText(sub,'Données de préentraînement, informations disponibles, sortie attendue.\nApport du mémoire : une grille critique et quatre protocoles à tester.',26);
setText(shapeById(s,'6'),'Conclusion',24);
}
// 11. Optional discussion support.
{
const n=11,s=slides[n-1];title(s,n,'Questions du jury');footer(s,n,'Annexe A');
grid(n,[['Quelle contribution personnelle ?','Une grille de comparaison des méthodes,\nune analyse critique des évaluations\net quatre protocoles contrôlés.'],['Quel modèle choisir ?','Préciser l’horizon, les covariables,\nla sortie attendue et le budget.\nComparer à une référence simple.'],['Comment vérifier le zero-shot ?','Figer le modèle et la référence.\nEnregistrer les prévisions avant les cibles,\npuis évaluer plusieurs domaines.'],['Pourquoi évaluer la dépendance ?','Les probabilités d’une somme élevée\nou d’un dépassement persistant dépendent\ndes relations entre horizons.']]);
}
// 12. Native backup table with criteria; full rejection rules in notes.
{
const n=12,s=slides[n-1];title(s,n,'Protocoles : critères proposés');footer(s,n,'Annexe B. Seuils proposés, aucun résultat observé');
at(s,'Les marges restent à justifier selon les tâches.',41.33,143,1197.33,50,28);
table(s,[['Hypothèse','Critère principal proposé','Contrôle essentiel'],['H1  Transfert','Gain prospectif moyen positif','Modèle et référence figés\nPrévisions horodatées'],['H2  Covariables','Gain ≥ 2 % avec les bonnes variables\nDégradation ≤ 5 % sous perturbation','Entrées correctes, retardées,\npermutées et contrôle aléatoire'],['H3  Adaptation','Surcroît de perte ≤ 2 % face au complet\nOubli ≤ 5 % et coût réduit','Même modèle initial\nEssais inclus dans le coût'],['H4  Trajectoires','Baisse moyenne du score de Brier\nface à l’indépendance','Marges et nombre de scénarios\nidentiques']],214,[245,540,412],389,22);
at(s,'Un intervalle qui recouvre zéro ou la marge pertinente reste non concluant.',41.33,619,1197.33,30,23);
}

const notes=JSON.parse(await fs.readFile(root+'/notes.json','utf8'));
const backup=JSON.parse(await fs.readFile(root+'/annexes_notes.json','utf8'));
for (let i=0;i<slides.length;i++){
 const n=(i<10?notes:backup).find(x=>x.slide===i+1);
 const body=n.script??JSON.stringify(n,null,2);
 slides[i].speakerNotes.textFrame.setText((i<10?`Durée conseillée : ${n.duration}. Repère : ${n.timeRange}.\n\n`:'Annexe facultative pour la discussion.\n\n')+body+`\n\nSource : ${n.sources}\nDocument : Memoire_Yoan_Kesraoui.pdf, fourni par Yoan Kesraoui.\n`);
}
const draft=root+'/draft/candidate.pptx';
await (await PresentationFile.exportPptx(p)).save(draft);
for(let i=0;i<slides.length;i++){
 await fs.writeFile(`${root}/draft/slide-${i+1}.png`,new Uint8Array(await (await p.export({slide:slides[i],format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(`${root}/draft/slide-${i+1}.json`,await (await slides[i].export({format:'layout'})).text());
}
await fs.writeFile(root+'/draft/inspect.ndjson',(await p.inspect({kind:'slide,textbox,image,table,notes',maxChars:200000})).ndjson);
const {finalizePresentation}=await import(pathToFileURL(SKILL_DIR+'/container_tools/artifact_tool_utils.mjs').href);
const finalPath=out+'/Soutenance_Yoan_Kesraoui_10_minutes.pptx';
const result=await finalizePresentation({workspaceDir,candidatePath:draft,finalPath,pythonExecutable:RUNTIME_PYTHON,integrityValidatorPath:SKILL_DIR+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL_DIR+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','6','--require-native-table-slide','12'],requiredNativeTableOwnerSlides:[6,12],fontPolicy:{basis:'reference',families:[family],referencePath:ref,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(ref)).digest('hex')},verifyArtifactToolImport:true,receiptPath:root+'/draft/validation.json'});
console.log(JSON.stringify(result));
