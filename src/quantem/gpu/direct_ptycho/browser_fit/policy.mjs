import {loadPyodide} from './pyodide.mjs';
export async function loadPolicy(){
 const url=new URL('.',import.meta.url);
 const py=await loadPyodide({indexURL:url.href});
 await py.loadPackage(['numpy','sqlalchemy','tqdm','pyyaml','packaging','typing-extensions','markupsafe']);
 const extras=await (await fetch(new URL('extras.json',url))).json();
 for(const name of extras)py.unpackArchive(await (await fetch(new URL(name,url))).arrayBuffer(),'wheel',{extractDir:py.FS.lookupPath('/lib/python3.13/site-packages').path});
 return py;
}
let runtime;
export async function fit(backend, cal, progress) {
 const py=await (runtime ??= loadPolicy().then(async py=>{
   py.FS.mkdirTree('/home/pyodide/policy');
   for(const name of ['policy/__init__.py','policy/higher_order.py','policy/fifth_order_parameters.py','policy/_gradient_refinement.py','solve.py']){
     const response=await fetch(new URL(name,import.meta.url));
     if(!response.ok)throw Error('Missing fit policy: '+name);
     py.FS.writeFile('/home/pyodide/'+name,await response.text());
   }
   return py;
 }).catch(error=>{runtime=undefined;throw error}));
 const scope=py.toPy({progress,wavelength:cal.wavelength_A,semiangle:cal.semiangle_mrad/1000,
   agreement:async (json,gradient)=>{
     const args=JSON.parse(json);
     if(gradient === "seed" || gradient === "batch") return JSON.stringify(await backend.fitBatch(args, gradient));
     if(!gradient)return JSON.stringify(await backend.modelAgreement(...args));
     const r=await backend.modelAgreementGradient(...args);
     return JSON.stringify([r.score,Array.from(r.gradient)]);
   }});
 try{return JSON.parse(await py.runPythonAsync('from solve import fit\nfit(agreement, progress, wavelength, semiangle)',{globals:scope}));}
 finally{scope.destroy();}
}
