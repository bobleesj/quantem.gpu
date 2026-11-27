import {fit} from "./policy.mjs";
let serial=0;
const pending=new Map();
function evaluate(args,gradient){
  return new Promise((resolve,reject)=>{
    const id=++serial;pending.set(id,{resolve,reject});
    postMessage({id,args,gradient});
  });
}
onmessage=async ({data})=>{
  if(data.id){
    const job=pending.get(data.id);pending.delete(data.id);
    if(data.error)job.reject(Error(data.error));else job.resolve(data.value);
    return;
  }
  try{
    const result=await fit({
      fitBatch:(args,mode)=>evaluate(args,mode),
      modelAgreement:(...args)=>evaluate(args,false),
      modelAgreementGradient:(...args)=>evaluate(args,true),
    },data.cal,status=>postMessage({status}));
    postMessage({result});
  }catch(error){postMessage({error:String(error)});}
};
