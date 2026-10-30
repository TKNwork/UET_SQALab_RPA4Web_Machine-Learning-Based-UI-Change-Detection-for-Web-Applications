import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const path=new URL('../results/dataset2_gt_review/index.html',import.meta.url);
const html=fs.readFileSync(path,'utf8');
class Element {
 constructor(tag,id=''){this.tag=tag;this.id=id;this.children=[];this.attrs={};this.style={};this.checked=['geometryOnly','show','numbers'].includes(id);this._value='';this.textContent='';this.listeners={};}
 set value(v){this._value=v}get value(){return this.tag==='select'&&!this._value?(this.children[0]?.value||''):this._value}
 append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes;if(this.tag==='select')this._value=''}
 setAttribute(k,v){this.attrs[k]=v}addEventListener(k,v){this.listeners[k]=v}
}
const els=new Map();for(const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))els.set(match[2],new Element(match[1],match[2]));
const document={getElementById:id=>{assert(els.has(id),id);return els.get(id)},createElement:tag=>new Element(tag),createElementNS:(_ns,tag)=>new Element(tag)};
const context=vm.createContext({document,console});vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
assert.equal(els.get('sample').children.length,166);
const samples=vm.runInContext('samples',context);assert.equal(samples.length,505);let boxes=0,outside=0;
for(const s of samples.filter(x=>x.count)){
 els.get('sample').value=s.id;els.get('sample').onchange();
 for(const side of ['before','after']){
  const data=s.sides[side];assert.equal(els.get(side+'Rows').children.length,data.boxes.length);
  for(const b of data.boxes){
   vm.runInContext(`selectGT(${JSON.stringify(side)},${b.original_index})`,context);
   const detail=JSON.parse(els.get('detail').textContent);assert.equal(detail.side,side);assert.equal(detail.original_index,b.original_index);
   els.get('focus').onclick();const v=els.get(side).attrs.viewBox.split(' ').map(Number);assert(v[0]<b.x&&v[1]<b.y&&v[0]+v[2]>b.x+b.w&&v[1]+v[3]>b.y+b.h);
   els.get('reset').onclick();assert.equal(els.get(side).attrs.viewBox,`0 0 ${data.width} ${data.height}`);boxes++;outside+=b.outside_viewport?1:0;
  }
 }
}
assert.equal(boxes,799);
els.get('geometryOnly').checked=false;els.get('geometryOnly').onchange();assert.equal(els.get('sample').children.length,505);
els.get('sample').value='501';els.get('sample').onchange();assert.equal(els.get('focus').disabled,true);
console.log(JSON.stringify({samples:505,geometry_samples:166,gt_bbox_selection_and_roi_checked:boxes,outside_viewport_annotations:outside,all_samples_filter:true,checks:'mock DOM interaction, not browser rendering'},null,2));
