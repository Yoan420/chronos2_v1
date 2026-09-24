import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const ROOT='C:/Users/BQ6757/chronos2_v1';
const BUILD=ROOT+'/.codex_presentation_report_additions_20260909';
const SOURCE=ROOT+'/deliverables/Presentation_Chronos2_FR_2026-08-26.pptx';
const REPORT=ROOT+'/runs/exports/2026-09-09/fr/nuclear_kalman/forecast_fr_2026-09-09_nuclear_kalman.html';
const FONT='Helvetica Neue', BLUE='#3D8DFF', SKY='#3F9FC9', BLACK='#000000', GREY='#545A64', GRID='#D7DAE0';
const {plots,payload}=JSON.parse(await fs.readFile(BUILD+'/report_data.json','utf8'));
const hourly=plots.find(p=>p.id.startsWith('hourly-comparison')).data;
const influence=plots.find(p=>p.id==='39d3e759-b6de-4039-8e44-717c80ac0faa').data[0];
const heat=plots.find(p=>p.id==='52c4b90e-0d75-417a-964f-5fed41567b3e').data[0];
function decode(v){
 if(Array.isArray(v))return v;
 if(v.dtype!=='f8')throw new Error('Unsupported array type');
 const buf=Buffer.from(v.bdata,'base64');
 const a=Array.from({length:buf.length/8},(_,i)=>buf.readDoubleLE(i*8));
 if(v.shape){const [rows,cols]=v.shape.split(',').map(Number);if(rows*cols!==a.length)throw new Error('Shape mismatch');return Array.from({length:rows},(_,i)=>a.slice(i*cols,(i+1)*cols));}
 return a;
}
const z=decode(heat.z), weights=decode(influence.x);
const daily=payload.records.filter(r=>r.zone==='FR'&&r.sample==='daily');
const day=daily.find(r=>r.period_key==='2026-09-09');
const recent=daily.slice(-7);
const n=daily.reduce((a,r)=>a+r.n,0);
const avg=key=>daily.reduce((a,r)=>a+r[key]*r.n,0)/n;
const fmt=(v,d=2)=>v.toFixed(d).replace('.',',');
const pres=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if(pres.slides.count!==3)throw new Error('Source slide count changed');
const originalSnapshots=pres.slides.items.map(s=>s.shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position})));

function tx(slide,text,x,y,w,h,size=21,opts={}){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text.set(text);
 s.text.style={typeface:FONT,fontSize:size,fill:BLACK,autoFit:'none',wrap:'square',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opts};
 return s;
}
function addSlide(title,subtitle){
 const slide=pres.slides.getItem(2).duplicate();
 slide.moveTo(pres.slides.count-1);
 const t=slide.shapes.items.find(s=>String(s.text??'').trim()==='Histo');
 if(!t)throw new Error('Source title not found');
 for(const s of [...slide.shapes.items])if(s.id!==t.id)slide.shapes.deleteById(s.id);
 t.text.set(title);
 t.text.style={typeface:FONT,fontSize:39,bold:true,fill:BLACK,autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};
 tx(slide,subtitle,41,111,1197,54,20,{fill:GREY});
 return slide;
}
function table(slide,values,x,y,width,height,colWidths,sz=18){
 const tb=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:colWidths,values});
 tb.styleOptions={headerRow:false,bandedRows:false,firstColumn:false};
 tb.borders.assign({fill:'#FFFFFF',width:2,style:'solid'});
 tb.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
  fill:'#FFFFFF',textStyle:{typeface:FONT,fontSize:sz,color:BLACK,alignment:'center',verticalAlignment:'middle',autoFit:'none'},margins:{left:5,right:5,top:5,bottom:5},anchor:'center',
 });
 return tb;
}
function note(slide,text){slide.speakerNotes.textFrame.setText(text+'\n\nSource : '+REPORT);slide.speakerNotes.setVisible(true);}
function footer(slide,text){tx(slide,text,41,672,1197,24,13,{fill:GREY});}
function mix(a,b,t){const aa=a.match(/\w\w/g).map(v=>parseInt(v,16)),bb=b.match(/\w\w/g).map(v=>parseInt(v,16));return '#'+aa.map((v,i)=>Math.round(v+(bb[i]-v)*Math.min(1,Math.max(0,t))).toString(16).padStart(2,'0')).join('');}

// Slide 4. Day headline, matched-price table, and a recent calendar strip.
const s4=addSlide('Prix moyens et comparaison à l’observé','Exemple du 9 septembre 2026 avec Chronos-2 + nucléaire + correcteur + Kalman');
const kpis=[
 [fmt(day.mean_price),'Prix moyen modèle P50',BLUE],
 [fmt(day.benchmark_mean_price),'Prix moyen Storm',BLACK],
 ['+'+fmt(day.mean_price-day.benchmark_mean_price),'Écart modèle − Storm',BLUE],
];
kpis.forEach(([value,label,color],i)=>{const x=41+i*404;tx(s4,value,x,177,375,68,54,{fill:color,bold:true});tx(s4,label+' (EUR/MWh)',x,246,375,34,18,{fill:GREY});});
tx(s4,'Prix moyens sur les mêmes heures',41,307,670,36,24,{bold:true});
const meanTable=table(s4,[
 ['Période','Observé','Modèle','Storm'],
 ['09/09/2026',fmt(day.observed_mean_price),fmt(day.mean_price),fmt(day.benchmark_mean_price)],
 ['365 jours¹',fmt(avg('observed_mean_price')),fmt(avg('mean_price')),fmt(avg('benchmark_mean_price'))],
],41,352,674,140,[224,150,150,150],20);
for(let c=0;c<4;c++){meanTable.getCell(0,c).fill='#EDEDED';meanTable.getCell(0,c).text.style={typeface:FONT,fontSize:18,bold:true,fill:c===2?BLUE:BLACK};}
for(let r=1;r<3;r++){
 meanTable.getCell(r,0).text.style={typeface:FONT,fontSize:18,fill:GREY};
 meanTable.getCell(r,2).text.style={typeface:FONT,fontSize:21,bold:true,fill:BLUE};
}
tx(s4,'Calendrier des écarts quotidiens',757,307,481,36,24,{bold:true});
const calendar=table(s4,[recent.map(r=>r.period_key.slice(8)+'/'+r.period_key.slice(5,7)),recent.map(r=>fmt(r.mean_price_absolute_error,1))],757,352,481,102,Array(7).fill(481/7),18);
for(let c=0;c<7;c++){
 calendar.getCell(0,c).text.style={typeface:FONT,fontSize:15,fill:GREY};
 calendar.getCell(1,c).fill=mix('FFF4EF','B2182B',recent[c].mean_price_absolute_error/30);
 calendar.getCell(1,c).text.style={typeface:FONT,fontSize:19,bold:true,fill:recent[c].mean_price_absolute_error>17?'#FFFFFF':BLACK};
}
tx(s4,'Le rouge fonce quand l’écart absolu entre\nle prix moyen du modèle et l’observé augmente.',757,471,481,67,18,{fill:GREY});
tx(s4,'Le 09/09 : écart absolu de 13,86 pour le modèle,\ncontre 1,72 pour Storm (EUR/MWh).',41,509,674,67,20);
tx(s4,'Dans Statistics : filtres jour, semaine et mois, puis classement du forecast le plus proche.\nUn prix moyen proche peut masquer des erreurs horaires qui se compensent.',41,593,1197,62,21,{fill:GREY});
footer(s4,'¹ Statistics : 10/09/2025 au 09/09/2026, 8 759 heures appariées. Storm sert uniquement à la comparaison.');
note(s4,`Le bandeau Prix moyens porte sur la journée civile locale du forecast, 24 heures appariées. Modèle ${day.mean_price}, Storm ${day.benchmark_mean_price}, écart ${day.mean_price-day.benchmark_mean_price} EUR/MWh. Le tableau Statistics ajoute l’observé ${day.observed_mean_price}.\nLe tableau filtre les périodes par jour, semaine ou mois et indique quel forecast a le prix moyen le plus proche de l’observé. Le calendrier mesure la valeur absolue de la différence entre la moyenne du modèle et celle de l’observé. Ici l’extrait affiche les sept derniers jours, avec une échelle de couleur 0 à 30 EUR/MWh.\nSur les ${n} heures communes, observé ${avg('observed_mean_price')}, modèle ${avg('mean_price')}, Storm ${avg('benchmark_mean_price')}. MAE horaire sur ces mêmes heures : modèle ${avg('mae')} contre Storm ${avg('benchmark_mae')}. La différence des moyennes peut être faible alors que les erreurs horaires restent importantes, car elles se compensent.\nLe modèle est plus proche en prix moyen dans 161 des 365 journées. Les cases plus rouges signalent un écart plus grand et non le sens de l’erreur. Les jours sans benchmark ne donnent pas de score comparatif.\nSections HTML : Prix moyens, Statistics — Prix moyens, Calendrier des écarts du modèle à l’observé. Les premières diapositives reprennent le support du 26 août, ces exemples proviennent du rapport du 9 septembre.`);

// Slide 5. Two actual views of the interactive hourly chart.
const s5=addSlide('Performance par heure locale','Le menu du rapport propose trois vues : MAE, prix moyens et biais signé');
const chartStyle={
 titlePlacement:'aboveChart',titleTextStyle:{typeface:FONT,fontSize:23,bold:true,fill:BLACK},
 hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:FONT,fontSize:17,fill:BLACK}},
 lineOptions:{grouping:'standard',smooth:false},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
 xAxis:{visible:true,textStyle:{typeface:FONT,fontSize:12,fill:GREY},line:{fill:GRID,width:1},majorGridlines:null},
};
const hourlyLabels=hourly[0].x.map(x=>x.slice(0,2));
function series(idx,name,color,dashed=false){return {name,values:hourly[idx].y,line:{fill:color,width:idx===0||idx===2?3.2:2.2,style:dashed?'dashed':'solid'},marker:{symbol:'none'}};}
s5.charts.add('line',{
 ...chartStyle,position:{left:41,top:182,width:582,height:350},title:'MAE horaire (EUR/MWh)',categories:hourlyLabels,
 series:[series(0,'Modèle',BLUE),series(1,'Storm',BLACK,true)],
 yAxis:{visible:true,min:0,max:18,majorUnit:3,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:15,fill:GREY},majorGridlines:{fill:GRID,width:0.8}},
});
s5.charts.add('line',{
 ...chartStyle,position:{left:657,top:182,width:581,height:350},title:'Prix moyens par heure (EUR/MWh)',categories:hourlyLabels,
 series:[series(4,'Observé',SKY),series(2,'Modèle',BLUE),series(3,'Storm',BLACK,true)],
 yAxis:{visible:true,min:0,max:120,majorUnit:30,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:15,fill:GREY},majorGridlines:{fill:GRID,width:0.8}},
});
tx(s5,'À 00 h, la MAE du modèle atteint 9,85,\ncontre 11,69 pour Storm.',41,557,582,70,23,{bold:true});
tx(s5,'À 12 h, elle atteint 12,04,\ncontre 10,71 pour Storm.',657,557,581,70,23,{bold:true});
tx(s5,'La MAE mesure la taille de l’erreur. Le biais signé indique la tendance à surestimer ou sous-estimer.',41,631,1197,32,20,{fill:GREY});
footer(s5,'France, 10/09/2025 au 09/09/2026. Chaque courbe utilise les mêmes 8 759 heures physiques appariées.');
note(s5,`Le graphique original Performance par heure locale dispose de trois modes : MAE, prix moyens et biais. Deux vues sont présentées simultanément ici. Chaque point regroupe les observations d’une même heure locale sur la période 10/09/2025–09/09/2026.\nMAE : moyenne des erreurs absolues. Biais : moyenne de la prévision moins le prix observé, positif pour la surestimation et négatif pour la sous-estimation. Les moyennes et les erreurs utilisent exactement les mêmes heures appariées, soit 8759 heures sur 8760 heures observées. L’heure locale 02h dispose de 364 points, les autres de 365.\nGain MAE = MAE Storm moins MAE modèle. Le modèle gagne dans 10 des 24 tranches horaires. Gain à 00h : ${hourly[1].y[0]-hourly[0].y[0]}, à 18h : ${hourly[1].y[18]-hourly[0].y[18]}, à 12h : ${hourly[1].y[12]-hourly[0].y[12]}, à 15h : ${hourly[1].y[15]-hourly[0].y[15]} EUR/MWh.\nLes données manquantes ne sont pas remplacées par zéro. Chaque heure physique compte une fois, en conservant les deux occurrences de 02h en automne.\nDonnées exactes des sept traces (heures 00h à 23h) :\n${JSON.stringify(hourly.map(t=>({name:t.name,values:t.y})))}`);

// Slide 6. Editable influence chart and editable heatmap table.
const s6=addSlide('Influence des variables dans la prévision amont','Shapley groupé sur Chronos-2 + nucléaire + correcteur, avant l’application de Kalman');
const labelMap={'Prix passés (contexte Chronos)':'Prix passés','Charge résiduelle FR':'Charge résid. FR','Charge résiduelle DE':'Charge résid. DE','Charge résiduelle BE':'Charge résid. BE','Charge résiduelle NL':'Charge résid. NL','Charge résiduelle ES':'Charge résid. ES','Production nucléaire prévue FR (GW)':'Nucléaire FR'};
s6.charts.add('bar',{
 position:{left:41,top:177,width:554,height:348},title:'Poids d’influence sur la journée (%)',titlePlacement:'aboveChart',titleTextStyle:{typeface:FONT,fontSize:23,bold:true,fill:BLACK},
 categories:influence.y.map(k=>labelMap[k]),series:[{name:'Influence Shapley (%)',values:weights,valuesFormatCode:'0.0',fill:BLUE,line:{fill:BLUE,width:0},points:weights.map((v,idx)=>({idx,fill:idx===0?SKY:BLUE}))}],
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:62},hasLegend:false,
 xAxis:{visible:true,textStyle:{typeface:FONT,fontSize:16,fill:BLACK},line:{fill:'#FFFFFF',width:0},majorGridlines:null},
 yAxis:{visible:true,min:0,max:65,majorUnit:20,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:14,fill:GREY},line:{fill:GRID,width:0.5},majorGridlines:{fill:GRID,width:0.5}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:16,bold:true,fill:BLUE}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
});
tx(s6,'Contributions par heure (EUR/MWh)',626,180,612,36,23,{bold:true});
const rowOrder=[6,1,0,4,2,3,5];
const heatValues=[['Heure locale',...Array.from({length:24},(_,i)=>i%3===0||i===23?String(i):'')],...rowOrder.map(i=>[labelMap[heat.y[i]],...Array(24).fill('')])];
const ht=table(s6,heatValues,626,226,612,273,[156,...Array(24).fill(19)],14);
ht.cells.block({row:0,column:1,rowCount:1,columnCount:24}).assign({margins:{left:0,right:0,top:0,bottom:0}});
for(let r=0;r<8;r++){
 ht.rows[r].height=r===0?28:35;
 ht.getCell(r,0).text.style={typeface:FONT,fontSize:14,fill:BLACK,alignment:'left'};
 for(let c=1;c<25;c++){
  const cell=ht.getCell(r,c);
  if(r===0){cell.text.style={typeface:FONT,fontSize:11,fill:GREY};continue;}
  const value=z[rowOrder[r-1]][c-1],max=heat.zmax;
  // Reproduce the report's diverging colour scale exactly.
  const t=(value+max)/(2*max),scale=heat.colorscale;
  let k=0;while(k<scale.length-2&&t>scale[k+1][0])k++;
  const rgbToHex=s=>s.match(/\d+/g).map(v=>Number(v).toString(16).padStart(2,'0')).join('');
  cell.fill=mix(rgbToHex(scale[k][1]),rgbToHex(scale[k+1][1]),(t-scale[k][0])/(scale[k+1][0]-scale[k][0]));
 }
}
tx(s6,'Bleu : baisse de la P50',782,506,230,29,16,{fill:'#2166AC'});
tx(s6,'Rouge : hausse de la P50',1016,506,222,29,16,{fill:'#B2182B'});
tx(s6,'Échelle symétrique : −43,2 à +43,2 EUR/MWh',782,532,456,23,14,{fill:GREY});
tx(s6,'136,24 + 27,09 = 163,33 EUR/MWh',41,555,1197,55,34,{fill:BLUE,bold:true});
tx(s6,'Référence historique + somme des contributions moyennes = P50 amont moyenne du jour',41,608,1197,32,20);
tx(s6,'Les % répartissent l’influence absolue. La carte montre le sens et l’intensité de chaque contribution.',41,643,1197,30,19,{fill:GREY});
footer(s6,'Sensibilité locale du 09/09/2026, relative à une médiane historique sur 56 jours. Ces influences ne mesurent pas un effet causal de marché.');
note(s6,`Les barres représentent le poids relatif Shapley, somme des contributions absolues d’un groupe sur les heures du jour divisée par la somme des contributions absolues de tous les groupes. Ce pourcentage ne représente pas une part du prix. Les sept groupes sont tous présents.\nLa carte thermique présente les contributions signées des 24 heures : positive, la variable relève le prix par rapport à la référence. Négative, elle le baisse. Les couleurs reprennent l’échelle symétrique originale de -${heat.zmax} à +${heat.zmax} EUR/MWh. Les cases restent éditables dans PowerPoint, les valeurs numériques exactes figurent ci-dessous.\nLa référence est une médiane par heure de la semaine des 56 jours historiques disponibles. Le modèle conserve ses paramètres appris et son calendrier. Shapley groupé par permutations recalcule le pipeline amont de bout en bout. Il explique Chronos-2 + nucléaire + correcteur, avant Kalman. Les états internes du filtre ne sont pas attribués aux variables.\nLes prix passés pèsent 57.198 % et apportent +21.027 EUR/MWh en moyenne. Le nucléaire pèse 1.085 %, avec une contribution moyenne -0.368 EUR/MWh. Ces valeurs décrivent la sensibilité de cette prévision à la référence et aux dépendances entre variables, sans identification d’un effet économique causal.\nRéférence moyenne 136.24 + contributions moyennes 27.09 = P50 amont moyenne 163.33 EUR/MWh. Dans ce rapport, l’ajustement Kalman moyen vaut 0.00.\nValeurs des contributions par groupe, heures 00 à 23 :\n${JSON.stringify(heat.y.map((label,i)=>({label,values:z[i]})))}\nPoids exacts : ${JSON.stringify(influence.y.map((label,i)=>({label,percent:weights[i]})))}`);

// Guard against unintended edits to the three user-supplied slides.
for(let i=0;i<3;i++){
 const snapshot=pres.slides.getItem(i).shapes.items.map(sh=>({text:String(sh.text??''),position:sh.position}));
 if(JSON.stringify(snapshot)!==JSON.stringify(originalSnapshots[i]))throw new Error('Original slide changed: '+(i+1));
}
if(pres.slides.count!==6)throw new Error('Expected exactly three additions');
await (await PresentationFile.exportPptx(pres)).save(BUILD+'/candidate.pptx');
console.log(JSON.stringify({candidate:BUILD+'/candidate.pptx',slides:pres.slides.count,hours:n,meanModel:avg('mean_price'),meanObserved:avg('observed_mean_price'),meanStorm:avg('benchmark_mean_price')},null,2));
