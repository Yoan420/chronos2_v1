// Regional launch contract: unavailable recipes cannot queue a production run.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('experiment_console/static/app.js', 'utf8');

async function main() {
  const elements = new Map(), listeners = {}, calls = [], storage = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      innerHTML:'', textContent:'', hidden:false, className:'', disabled:false,
      addEventListener(){},querySelector:selector=>element(id+' '+selector), classList:{toggle(){}},
      setAttribute(){}, removeAttribute(){}, contains(){return false;},
      showModal(){this.open=true;}, close(){this.open=false;}, reportValidity(){return true;},
    });
    return elements.get(id);
  };
  let gate={delivery_day:'2026-10-01',countries:['FR'],ready:false,recipe_status:'pending_backtest',blockers:['Backtest à valider.']};
  const evaluationGate={operation:'evaluate',ready:true,recipe_status:'pending_backtest',blockers:[]};
  let runs=[];
  let regionalArtifacts=[];
  let evaluationArtifacts=[];
  const evaluationReceipt={confirmation_first_day:'2026-05-01',stop_day_exclusive:'2026-09-23',countries:{
    FR:{selected:'absolute',confirmation:{rmse:1.25,storm_rmse:2.5,strict_win_rate:.6},negative_confirmation:{brier:.12}},
  }};
  let previewCount=0;
  const context=vm.createContext({
    document:{querySelector:element,querySelectorAll:()=>[],addEventListener(type,fn){(listeners[type]??=[]).push(fn);}},
    window:{addEventListener(){}},navigator:{},location:{hash:'#regional'},
    history:{replaceState(_a,_b,hash){context.location.hash=hash;}},
    sessionStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    crypto:{randomUUID:()=>`key-${++previewCount}`},setTimeout:()=>1,clearTimeout(){},setInterval(){},
    renderArchitecture:async()=>{},
    fetch:async(path,options={})=>{
      calls.push({path,options});
      let value;
      if(path==='/api/primary-results')value={days:[],warnings:[]};
      else if(path==='/api/primary-run')value={available:true,defaults:{delivery_day:'2026-10-01'},active:null,latest:null};
      else if(path==='/api/bootstrap')value={token:'token',catalog:[{id:'nyx_regional_cpu',parameters:[{name:'delivery_day',default:'2026-10-01'}]}]};
      else if(path.startsWith('/api/regional-preflight'))value=gate;
      else if(path==='/api/regional-evaluation-preflight')value=evaluationGate;
      else if(path==='/api/runs')value={runs};
      else if(path==='/api/preview')value={id:JSON.parse(options.body).adapter_id==='nyx_regional_cpu_backtest'?'eval-plan':'plan-1'};
      else if(path==='/api/launch'){
        const evaluation=JSON.parse(options.body).plan_id==='eval-plan';
        value={id:evaluation?'eval-run':'run-1',adapter_id:evaluation?'nyx_regional_cpu_backtest':'nyx_regional_cpu',status:'queued',
          delivery_day:evaluation?null:'2026-10-01',countries:evaluation?[]:['FR'],created_at:'2026-09-30T10:00:00Z'};
        runs=[value,...runs];
      }else if(path==='/api/runs/run-1'||path==='/api/runs/eval-run')value={...runs.find(run=>path.endsWith(run.id)),
        artifacts:path==='/api/runs/run-1'?regionalArtifacts:evaluationArtifacts,output_dir:'runs/test/outputs'};
      else if(path==='/api/runs/eval-run/artifact?path=results%2Fbacktest_receipt.json')value=evaluationReceipt;
      else throw Error(`Unexpected ${path}`);
      return {ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(value))};
    },
  });
  vm.runInContext(source,context);
  await new Promise(resolve=>setImmediate(resolve));
  const evaluate=code=>vm.runInContext(code,context);
  assert.equal(evaluate('state.route'),'regional');
  assert.equal(evaluate('state.regional.day'),'2026-10-01');
  assert.equal(element('#regional-launch').disabled,true);
  assert.ok(element('#regional-check').innerHTML.includes('Backtest à valider.'));
  assert.equal(element('#regional-evaluate').disabled,false);
  await evaluate('launchRegional()');
  assert.ok(!calls.some(call=>call.path==='/api/preview'||call.path==='/api/launch'));

  await evaluate('launchRegionalEvaluation()');
  assert.equal(calls.filter(call=>call.path==='/api/preview').length,1);
  assert.equal(JSON.parse(calls.find(call=>call.path==='/api/preview').options.body).adapter_id,'nyx_regional_cpu_backtest');
  assert.equal(element('#regional-evaluate').disabled,true);
  assert.equal(element('#regional-launch').disabled,true);
  runs[0].status='failed';
  evaluationArtifacts=[{path:'results/backtest_receipt.json',name:'backtest_receipt.json'}];
  await evaluate('refreshRegional()');
  assert.ok(element('#regional-output').innerHTML.includes('Scores du backtest'));
  assert.ok(element('#regional-output').innerHTML.includes('RMSE Storm'));
  assert.ok(element('#regional-output').innerHTML.includes('1,25'));
  assert.ok(element('#regional-output').innerHTML.includes('backtest_receipt.json'));
  runs[0].status='succeeded';
  gate={...gate,ready:true,recipe_status:'validated',blockers:[]};
  await evaluate('refreshRegional()');
  await evaluate('checkRegional()');

  assert.equal(element('#regional-launch').disabled,false);
  await evaluate('launchRegional()');
  assert.equal(calls.filter(call=>call.path==='/api/preview').length,2);
  const launch=calls.filter(call=>call.path==='/api/launch').at(-1);
  assert.equal(launch.options.body,'{"plan_id":"plan-1","idempotency_key":"key-2"}');
  assert.equal(element('#regional-launch').disabled,true,'an active run locks another request');
  assert.ok(element('#regional-output').innerHTML.includes('En attente'));
  assert.equal(storage.has('nyx-regional-intent'),false);
  runs[0].status='succeeded';
  regionalArtifacts=['csv','html'].map(suffix=>({
    path:`zones/FR/forecast_fr_2026-10-01_nyx_regional_cpu.${suffix}`,
    name:`forecast_fr_2026-10-01_nyx_regional_cpu.${suffix}`,
  }));
  await evaluate('refreshRegional()');
  assert.ok(element('#regional-output').innerHTML.includes('forecast_fr_2026-10-01_nyx_regional_cpu.csv ↓'));
  assert.ok(element('#regional-output').innerHTML.includes('Ouvrir le rapport'));
  assert.ok(element('#regional-output').innerHTML.includes('zones%2FFR%2Fforecast_fr_2026-10-01_nyx_regional_cpu.csv'));
  console.log('NYX regional frontend contracts passed: evaluation, validation gate, forecast launch, active lock.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
