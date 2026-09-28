// Annual controls must use their own qualification gate and four-country job.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

async function main(){
  const elements=new Map(), calls=[], storage=new Map();
  const element=id=>{
    if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',disabled:false,
      value:id==='#annual-pipeline-day'?'2026-10-01':'',
      reportValidity:()=>true,addEventListener(){},classList:{toggle(){}},
      setAttribute(){},removeAttribute(){},focus(){}});
    return elements.get(id);
  };
  let gate={delivery_day:'2026-10-01',countries:['FR','DE','NL','BE'],
    can_capture:false,can_prepare:true,can_request_forecast:false,ready:false,
    activation_error:'Qualification complète absente',source_states:{saturn:'MISSING'}};
  let runs=[], lastParameters;
  const context=vm.createContext({document:{querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
    window:{addEventListener(){}},navigator:{},location:{hash:'#regional'},
    sessionStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    crypto:{randomUUID:()=> 'annual-key'},setTimeout:()=>1,clearTimeout(){},setInterval(){},
    fetch:async(path,options={})=>{
      calls.push({path,options});let payload;
      if(path.startsWith('/api/annual-pipeline-preflight?'))payload=gate;
      else if(path==='/api/bootstrap')payload={token:'token',catalog:[]};
      else if(path==='/api/annual-publications')payload={days:[{delivery_day:'2026-09-30',countries:[{zone:'DE',hours:24,report_available:true}]}],warnings:[]};
      else if(path==='/api/preview'){
        const request=JSON.parse(options.body);assert.equal(request.adapter_id,'nyx_annual_pipeline');
        assert.equal(request.config_id,'annual_cpu_four_countries');lastParameters=request.parameters;
        payload={id:'annual-plan'};
      }else if(path==='/api/launch'){
        assert.equal(JSON.parse(options.body).plan_id,'annual-plan');
        payload={id:'annual-run',adapter_id:'nyx_annual_pipeline',status:'queued',
          request:{parameters:lastParameters},countries:['FR','DE','NL','BE'],delivery_day:'2026-10-01'};
        runs=[payload];
      }else if(path==='/api/runs')payload={runs};
      else if(path==='/api/runs/annual-run')payload={...runs[0],artifacts:[],output_dir:'runs/annual-output'};
      else if(path==='/api/runs/annual-run/logs')payload={available:true,text:'Sources enregistrées'};
      else throw Error('Unexpected request '+path);
      return {ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(payload))};
    }});
  const source=fs.readFileSync('experiment_console/static/app.js','utf8')
    .replace('void refresh();\nsetInterval(()=>void refresh(),5000);','');
  vm.runInContext(source,context);
  const run=code=>vm.runInContext(code,context);
  run("state.route='regional';state.regional.day='2026-10-01';renderRegional()");
  assert.ok(element('#main').innerHTML.includes('FR / DE / NL / BE'));
  await run('checkAnnualPipeline()');
  assert.equal(element('#annual-prepare').disabled,false);
  assert.equal(element('#annual-capture').disabled,true);
  assert.equal(element('#annual-forecast').disabled,true);
  await run("launchAnnualPipeline('forecast')");
  assert.equal(calls.filter(c=>c.path==='/api/preview').length,0);
  await run("launchAnnualPipeline('prepare')");
  assert.deepEqual(lastParameters,{delivery_day:'2026-10-01',action:'prepare'});
  assert.equal(element('#annual-prepare').disabled,true,'active work locks new jobs');
  assert.ok(element('#regional-runs').innerHTML.includes('Préparation CPU'));
  assert.ok(element('#annual-publications').innerHTML.includes('DE · rapport'));
  assert.ok(element('#annual-publications').innerHTML.includes('format=csv&amp;download=1'));
  runs[0].status='succeeded';
  await run('refreshRegional()');
  assert.ok(element('#regional-output').innerHTML.includes('Préparation des quatre modèles terminée'));
  assert.ok(!element('#regional-output').innerHTML.includes('aucun résultat consultable'));
  assert.ok(element('#regional-output').innerHTML.includes('Sources enregistrées'));
  gate={...gate,can_prepare:false,can_capture:true};
  await run("launchAnnualPipeline('capture')");
  assert.equal(lastParameters.action,'capture');
  runs[0].status='succeeded';await run('refreshRegional()');
  assert.ok(element('#regional-output').innerHTML.includes('Captures quotidiennes enregistrées'));
  gate={...gate,can_request_forecast:true,can_capture:false,activation_error:null};
  await run("launchAnnualPipeline('forecast')");
  assert.equal(lastParameters.action,'forecast');
  assert.ok(element('#regional-runs').innerHTML.includes('Prévision annuelle'));
  assert.equal(storage.size,0);
  runs=[];await run('refreshRegional()');
  const launches=calls.filter(c=>c.path==='/api/launch').length;
  element('#annual-pipeline-day').reportValidity=()=>false;
  await run("launchAnnualPipeline('forecast')");
  assert.equal(calls.filter(c=>c.path==='/api/launch').length,launches,'invalid date cannot reuse a previous gate');
  console.log('Annual frontend contracts passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
