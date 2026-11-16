/** Run the numerical search on this device, keeping the UI thread available for GPU work. */
export function fit(backend, cal, progress, prepared = Promise.resolve()) {
  const worker = new Worker(new URL("fit-worker.mjs", import.meta.url), {type:"module"});
  return new Promise((resolve,reject)=>{
    const fail = message => {worker.terminate();reject(new Error(message));};
    prepared.catch(error => fail(String(error)));
    worker.onerror = e => fail(e.message);
    worker.onmessage = async ({data})=>{
      if(data.error){fail(data.error);return;}
      if(data.result){worker.terminate();resolve(data.result);return;}
      if(data.status){progress(data.status);return;}
      try {
        await prepared;
        if(data.gradient === "seed" || data.gradient === "batch"){
          const value=await backend.evaluateFitBatch(data.args,data.gradient === "seed");
          worker.postMessage({id:data.id,value});return;
        }
        const value = await backend[data.gradient ? "modelAgreementGradient" : "modelAgreement"](...data.args);
        worker.postMessage({id:data.id, value:data.gradient ? {score:value.score,gradient:Array.from(value.gradient)} : value});
      } catch(error){worker.postMessage({id:data.id,error:String(error)});}
    };
    worker.postMessage({cal:{wavelength_A:cal.wavelength_A,semiangle_mrad:cal.semiangle_mrad}});
  });
}
