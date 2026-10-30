// Check all 97 sample selections and embedded crops without network/browser.
import fs from 'node:fs';
import vm from 'node:vm';
const input=process.argv[2];
if(!input)throw Error('Supply the inline fragment path');
const fragment=fs.readFileSync(input,'utf8');
if(Buffer.byteLength(fragment)>=1_000_000)throw Error('Too large');
if(/<!doctype|<html|<body/i.test(fragment))throw Error('Expected an inline fragment');
const script=fragment.match(/<script>([\s\S]*?)<\/script>/)[1];
class Node {
  constructor(){this.children=[];this.attrs={};this.textContent='';this.listeners={};}
  append(...items){this.children.push(...items);}
  setAttribute(k,v){this.attrs[k]=v;}
  addEventListener(k,v){this.listeners[k]=v;}
}
const elements=Object.fromEntries(['grid','selection','description','before-image','after-image','before-kind','after-kind','before-text','after-text','status','details'].map(key=>[key,new Node()]));
const root={querySelector:query=>elements[query.match(/data-catalog="([^"]+)"/)[1]]};
const window={addEventListener:()=>{},openai:{setWidgetState:()=>Promise.resolve()}};
vm.runInNewContext(script,{document:{getElementById:id=>{if(id!=='content-change-catalog-97')throw Error('Wrong root');return root;},createElement:()=>new Node()},window,console});
if(elements.grid.children.length!==97)throw Error('Not all content samples');
let evaluated=0,notEvaluated=0;
for(const button of elements.grid.children){
  button.listeners.click();
  const detail=JSON.parse(elements.details.textContent);
  if(detail.id!==button.textContent)throw Error('Wrong selected sample');
  if(button.attrs['aria-pressed']!=='true')throw Error('Missing selection indication');
  if(!elements.selection.textContent.startsWith(detail.id+' ·'))throw Error('Wrong summary');
  if(elements.selection.textContent.includes('Đã chấm:'))evaluated++;else notEvaluated++;
  for(const side of ['before','after']){
    if(!elements[side+'-image'].src.startsWith('data:image/webp;base64,'))throw Error('Missing embedded image');
    const image=Buffer.from(elements[side+'-image'].src.split(',')[1],'base64');
    if(image.subarray(0,4).toString()!=='RIFF')throw Error('Invalid WebP');
    if(!elements[side+'-text'].textContent)throw Error('Missing target text/fallback label');
  }
}
if(evaluated!==6||notEvaluated!==91)throw Error('Confused GT status');
console.log(JSON.stringify({samples:97,evaluated,not_evaluated:notEvaluated,embedded_images:194,bytes:Buffer.byteLength(fragment),all_selections_pass:true}));
