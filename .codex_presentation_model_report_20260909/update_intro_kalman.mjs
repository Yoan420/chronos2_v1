import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const ROOT='C:/Users/BQ6757/chronos2_v1';
const SOURCE=ROOT+'/deliverables/Presentation_Chronos2_FR_2026-08-26_completee_rapport_2026-09-09.pptx';
const BUILD=ROOT+'/.codex_presentation_intro_kalman_20260909';
const FONT='Helvetica Neue',BLUE='#3D8DFF',SKY='#6DCBF4',KALMAN='#2468B8',BLACK='#000000',GREY='#545A64';
const pres=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if(pres.slides.count!==6)throw new Error('Expected six source slides');
const untouched=pres.slides.items.slice(2).map(s=>s.shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position})));
const [pipeline,models]=pres.slides.items;
function text(shape,value,size=20,opts={}){
 shape.text.set(value);
 shape.text.style={typeface:FONT,fontSize:size,fill:BLACK,autoFit:'none',wrap:'square',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opts};
 return shape;
}
function move(shape,left,top,width,height){shape.position={...shape.position,left,top,width,height};return shape;}
function tx(slide,value,left,top,width,height,size=20,opts={}){
 return text(slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}}),value,size,opts);
}
function box(slide,name,left,top,width,height,fill){return slide.shapes.add({name,geometry:'roundRect',position:{left,top,width,height},fill,line:{fill,width:0}});}
function circle(slide,name,left,top,diameter,fill){return slide.shapes.add({name,geometry:'ellipse',position:{left,top,width:diameter,height:diameter},fill,line:{fill,width:0}});}
function notes(slide,body){slide.speakerNotes.textFrame.setText(body+'\n\nSources : deliverables/Note_explicative_Kalman.txt ; chronos2_hourly/kalman_residual.py ; config/kalman_operational.yaml ; config/nuclear_forecast.yaml.');slide.speakerNotes.setVisible(true);}

// Insert the Kalman step into the existing editable pipeline.
const pShapes=[...pipeline.shapes.items];
const stageIndices=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]];
const existing=stageIndices.map(indices=>indices.map(i=>pShapes[i]));
const kalmanStage=[
 tx(pipeline,'ADAPTATION',648,225,184,30,15,{bold:true,fill:KALMAN}),
 circle(pipeline,'node-kalman',648,294,26,KALMAN),
 tx(pipeline,'Kalman',648,354,184,38,23,{bold:true}),
 tx(pipeline,'Suit le biais récent\nCorrige si le gain\npassé est suffisant',648,404,184,125,17,{fill:GREY}),
];
const allStages=[existing[0],existing[1],existing[2],kalmanStage,existing[3],existing[4]];
const xPositions=[60,256,452,648,844,1040];
allStages.forEach((stage,i)=>{
 const [label,node,title,body]=stage,x=xPositions[i];
 move(label,x,225,184,30);
 move(node,x,294,26,26);
 move(title,x,354,184,38);
 move(body,x,404,184,140);
});
move(pShapes[0],60,307,1143,0.030026246719160106);
text(existing[4][2],'Évaluation',23,{bold:true});
text(existing[3][3],'23, 24 ou 25 heures\nP10 ≤ P50 ≤ P90\nAbsence de fuite future',17,{fill:GREY});
tx(pipeline,'Pipeline de prévision avec filtre de Kalman',41,36,1197,66,39,{bold:true});
tx(pipeline,'Kalman intervient après le correcteur résiduel, avant les contrôles et la publication.',41,111,1197,58,21,{fill:GREY});
tx(pipeline,'À chaque heure, la correction Kalman décale ensemble P10, P50 et P90.',60,603,1150,40,23,{fill:KALMAN,bold:true});
notes(pipeline,'Le filtre de Kalman intervient après la prévision corrigée résiduellement et avant sa publication. Il estime une erreur persistante à partir des seules erreurs déjà observées. La gouvernance choisit chaque jour un filtre et un poids de correction sur les 60 derniers jours disponibles. En l’absence de gain suffisant, le poids vaut zéro et la prévision amont reste identique.\nLe filtre reçoit uniquement des données disponibles avant la journée à prévoir. Son état est figé pour toutes les heures de cette journée et ne peut assimiler ses observations qu’après la prévision. Les journées de changement d’heure comportent 23 ou 25 heures physiques.\nLes charges résiduelles des cinq pays sont les fondamentaux du pipeline standard. Dans la variante nucléaire utilisée pour illustrer le rapport du 09/09/2026, la prévision de production nucléaire FR constitue une entrée supplémentaire. Storm intervient uniquement dans la comparaison des résultats.');

// Reflow the existing model explanation into three columns.
const m=[...models.shapes.items];
const keep=new Set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21].map(i=>m[i].id));
for(const sh of m)if(!keep.has(sh.id))models.shapes.deleteById(sh.id);
text(move(m[0],41,111,1197,56),'Chronos-2 fournit la base, le correcteur apprend les erreurs récurrentes et Kalman suit les biais récents.',20,{fill:GREY});
tx(models,'Chronos-2, correcteur résiduel et Kalman',41,36,1197,66,39,{bold:true});
move(m[1],41,181,378,429);m[1].fill='#F2F2F2';
move(m[2],450,181,378,429);m[2].fill='#F2F2F2';
const kalmanCard=box(models,'kalman-card',860,181,378,429,'#F2F2F2');kalmanCard.sendToBack();
// Chronos-2 column.
move(m[3],64,210,32,32);
text(move(m[4],75,216,15,24),'1',16,{bold:true,fill:'#FFFFFF'});
text(move(m[5],110,206,286,48),'Chronos-2',28,{bold:true});
text(move(m[6],64,273,330,55),'Prévision de base',22,{bold:true,fill:BLUE});
text(move(m[7],64,337,330,147),'Jusqu’à 2 048 h de prix passés.\n\nCalendrier et charges résiduelles\ndes cinq pays, connus à D−1.\n\nProduit une distribution de prix.',18);
move(m[8],64,508,330,0.030026246719160106);
for(let i=0;i<3;i++){
 text(move(m[9+i],70+i*111,535,88,30),['P10','P50','P90'][i],19,{bold:true,fill:i===1?BLUE:'#3F9FC9'});
 text(move(m[12+i],70+i*111,573,100,25),['bas','central','haut'][i],15,{fill:GREY});
}
// Residual corrector column.
move(m[15],473,210,32,32);
text(move(m[16],484,216,15,24),'2',16,{bold:true});
text(move(m[17],519,206,286,48),'Correcteur résiduel',25,{bold:true});
text(move(m[18],473,273,330,55),'Apprentissage des erreurs',22,{bold:true,fill:BLUE});
text(move(m[19],473,337,330,165),'Cible : prix réel − P50 Chronos-2\n\nCatBoost et HistGradientBoosting\najoutent des arbres successifs\npour réduire les erreurs.',18);
move(m[20],473,518,330,76);
text(move(m[21],484,535,308,55),'CatBoost 50 %\nHistGradientBoosting 50 %',18,{bold:true,alignment:'center'});
// Kalman column, using the same card and numbering style.
circle(models,'stage-three',883,210,32,KALMAN);
tx(models,'3',894,216,15,24,16,{bold:true,fill:'#FFFFFF'});
tx(models,'Filtre de Kalman',929,206,286,48,28,{bold:true});
tx(models,'Adaptation au biais récent',883,273,330,55,22,{bold:true,fill:KALMAN});
tx(models,'Estime le biais à partir des erreurs\ndéjà observées.\n\nChoisit le filtre et son poids\nsur les 60 derniers jours.\n\nSans gain suffisant, conserve\nla prévision amont.',883,337,330,199,18);
tx(models,'Même décalage sur\nP10, P50 et P90',883,551,330,56,19,{bold:true,fill:KALMAN});
tx(models,'Exemple illustratif : 150 + 8 + 3 = 161 EUR/MWh',41,645,1197,43,28,{bold:true,fill:BLUE});
notes(models,'Trois étages complémentaires : Chronos-2 construit la prévision de base. Le correcteur résiduel, moyenne de CatBoost et HistGradientBoosting, apprend les erreurs récurrentes de la médiane Chronos-2. Le Kalman estime ensuite un biais persistant dans la prévision corrigée, en utilisant seulement les erreurs passées.\nL’exemple est fictif et pédagogique : P50 Chronos-2 150 EUR/MWh, correction résiduelle +8, correction Kalman appliquée +3, P50 finale 161 EUR/MWh. La correction Kalman peut valoir zéro. Le même décalage est ajouté à P10, P50 et P90 à une heure donnée, ce qui conserve leur ordre et la largeur de l’intervalle.\nParamètres actuels : refit journalier sur les 365 jours précédents. Gouvernance sur 60 jours, avec au moins 14 jours. Poids de 0 à 1 par pas de 0,05. Le gain de MAE doit atteindre le maximum de 0,05 EUR/MWh et 0,5 % de la MAE amont. Les filtres candidats suivent un biais, un profil horaire, des variables de marché ou une échelle de correction linéaire/non linéaire. Le décalage est borné à ±20 EUR/MWh.\nL’état est gelé avant la journée de prévision. Les observations de cette journée ne peuvent influencer que les journées suivantes.\nLa slide historique initiale conserve les scores du support d’origine : elle ne mesure pas le gain marginal de Kalman. Les nouveaux visuels de rapport restent sur leur périmètre déclaré du 09/09/2026.');

for(let i=2;i<6;i++){
 const current=pres.slides.getItem(i).shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position}));
 if(JSON.stringify(current)!==JSON.stringify(untouched[i-2]))throw new Error('Unrelated slide changed: '+(i+1));
}
await fs.mkdir(BUILD,{recursive:true});
await (await PresentationFile.exportPptx(pres)).save(BUILD+'/candidate.pptx');
console.log(JSON.stringify({candidate:BUILD+'/candidate.pptx',slides:pres.slides.count,changedSlides:[1,2]},null,2));
