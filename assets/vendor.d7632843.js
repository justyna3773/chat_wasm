let f;const l=new Array(128).fill(void 0);l.push(void 0,null,!0,!1);function r(n){return l[n]}let S=l.length;function M(n){n<132||(l[n]=S,S=n)}function a(n){const e=r(n);return M(n),e}const L=new TextDecoder("utf-8",{ignoreBOM:!0,fatal:!0});L.decode();let y=null;function T(){return(y===null||y.byteLength===0)&&(y=new Uint8Array(f.memory.buffer)),y}function d(n,e){return L.decode(T().subarray(n,n+e))}function o(n){S===l.length&&l.push(l.length+1);const e=S;return S=l[e],l[e]=n,e}let g=0;const A=new TextEncoder("utf-8"),R=typeof A.encodeInto=="function"?function(n,e){return A.encodeInto(n,e)}:function(n,e){const t=A.encode(n);return e.set(t),{read:n.length,written:t.length}};function m(n,e,t){if(t===void 0){const i=A.encode(n),p=e(i.length);return T().subarray(p,p+i.length).set(i),g=i.length,p}let _=n.length,c=e(_);const b=T();let u=0;for(;u<_;u++){const i=n.charCodeAt(u);if(i>127)break;b[c+u]=i}if(u!==_){u!==0&&(n=n.slice(u)),c=t(c,_,_=u+n.length*3);const i=T().subarray(c+u,c+_);u+=R(n,i).written}return g=u,c}function P(n){return n==null}let h=null;function s(){return(h===null||h.byteLength===0)&&(h=new Int32Array(f.memory.buffer)),h}function O(n){const e=typeof n;if(e=="number"||e=="boolean"||n==null)return`${n}`;if(e=="string")return`"${n}"`;if(e=="symbol"){const c=n.description;return c==null?"Symbol":`Symbol(${c})`}if(e=="function"){const c=n.name;return typeof c=="string"&&c.length>0?`Function(${c})`:"Function"}if(Array.isArray(n)){const c=n.length;let b="[";c>0&&(b+=O(n[0]));for(let u=1;u<c;u++)b+=", "+O(n[u]);return b+="]",b}const t=/\[object ([^\]]+)\]/.exec(toString.call(n));let _;if(t.length>1)_=t[1];else return toString.call(n);if(_=="Object")try{return"Object("+JSON.stringify(n)+")"}catch{return"Object"}return n instanceof Error?`${n.name}: ${n.message}
${n.stack}`:_}function G(n,e,t,_){const c={a:n,b:e,cnt:1,dtor:t},b=(...u)=>{c.cnt++;const i=c.a;c.a=0;try{return _(i,c.b,...u)}finally{--c.cnt===0?f.__wbindgen_export_2.get(c.dtor)(i,c.b):c.a=i}};return b.original=c,b}function C(n,e,t){f._dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h379b29d191c696a5(n,e,o(t))}function U(n,e,t){f._dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__ha85f54a9d7f102e5(n,e,o(t))}let x=null;function V(){return(x===null||x.byteLength===0)&&(x=new Float32Array(f.memory.buffer)),x}function D(n,e){const t=e(n.length*4);return V().set(n,t/4),g=n.length,t}function k(n,e){const t=e(n.length*1);return T().set(n,t/1),g=n.length,t}function F(n,e){if(!(n instanceof e))throw new Error(`expected instance of ${e.name}`);return n.ptr}function w(n,e){try{return n.apply(this,e)}catch(t){f.__wbindgen_exn_store(o(t))}}let B=null;function v(){return(B===null||B.byteLength===0)&&(B=new Uint32Array(f.memory.buffer)),B}function E(n,e){return v().subarray(n/4,n/4+e)}function q(n,e,t,_){f.wasm_bindgen__convert__closures__invoke2_mut__h3af64c9774cbde22(n,e,o(t),o(_))}class I{static __wrap(e){const t=Object.create(I.prototype);return t.ptr=e,t}__destroy_into_raw(){const e=this.ptr;return this.ptr=0,e}free(){const e=this.__destroy_into_raw();f.__wbg_input_free(e)}constructor(){const e=f.input_new();return I.__wrap(e)}insert(e,t){const _=m(e,f.__wbindgen_malloc,f.__wbindgen_realloc),c=g,b=D(t,f.__wbindgen_malloc),u=g;f.input_insert(this.ptr,_,c,b,u)}}class W{static __wrap(e){const t=Object.create(W.prototype);return t.ptr=e,t}__destroy_into_raw(){const e=this.ptr;return this.ptr=0,e}free(){const e=this.__destroy_into_raw();f.__wbg_session_free(e)}static fromBytes(e){const t=k(e,f.__wbindgen_malloc),_=g,c=f.session_fromBytes(t,_);return a(c)}run(e){F(e,I);const t=f.session_run(this.ptr,e.ptr);return a(t)}}class j{static __wrap(e){const t=Object.create(j.prototype);return t.ptr=e,t}__destroy_into_raw(){const e=this.ptr;return this.ptr=0,e}free(){const e=this.__destroy_into_raw();f.__wbg_sessionerror_free(e)}toString(){try{const _=f.__wbindgen_add_to_stack_pointer(-16);f.sessionerror_toString(_,this.ptr);var e=s()[_/4+0],t=s()[_/4+1];return d(e,t)}finally{f.__wbindgen_add_to_stack_pointer(16),f.__wbindgen_free(e,t)}}}async function $(n,e){if(typeof Response=="function"&&n instanceof Response){if(typeof WebAssembly.instantiateStreaming=="function")try{return await WebAssembly.instantiateStreaming(n,e)}catch(_){if(n.headers.get("Content-Type")!="application/wasm")console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n",_);else throw _}const t=await n.arrayBuffer();return await WebAssembly.instantiate(t,e)}else{const t=await WebAssembly.instantiate(n,e);return t instanceof WebAssembly.Instance?{instance:t,module:n}:t}}function z(){const n={};return n.wbg={},n.wbg.__wbindgen_is_string=function(e){return typeof r(e)=="string"},n.wbg.__wbg_session_new=function(e){const t=W.__wrap(e);return o(t)},n.wbg.__wbg_sessionerror_new=function(e){const t=j.__wrap(e);return o(t)},n.wbg.__wbindgen_object_drop_ref=function(e){a(e)},n.wbg.__wbindgen_error_new=function(e,t){const _=new Error(d(e,t));return o(_)},n.wbg.__wbindgen_bigint_from_i64=function(e){return o(e)},n.wbg.__wbindgen_number_new=function(e){return o(e)},n.wbg.__wbindgen_object_clone_ref=function(e){const t=r(e);return o(t)},n.wbg.__wbindgen_string_new=function(e,t){const _=d(e,t);return o(_)},n.wbg.__wbindgen_is_object=function(e){const t=r(e);return typeof t=="object"&&t!==null},n.wbg.__wbindgen_string_get=function(e,t){const _=r(t),c=typeof _=="string"?_:void 0;var b=P(c)?0:m(c,f.__wbindgen_malloc,f.__wbindgen_realloc),u=g;s()[e/4+1]=u,s()[e/4+0]=b},n.wbg.__wbg_set_841ac57cff3d672b=function(e,t,_){r(e)[a(t)]=a(_)},n.wbg.__wbg_new_abda76e883ba8a5f=function(){const e=new Error;return o(e)},n.wbg.__wbg_stack_658279fe44541cf6=function(e,t){const _=r(t).stack,c=m(_,f.__wbindgen_malloc,f.__wbindgen_realloc),b=g;s()[e/4+1]=b,s()[e/4+0]=c},n.wbg.__wbg_error_f851667af71bcfc6=function(e,t){try{console.error(d(e,t))}finally{f.__wbindgen_free(e,t)}},n.wbg.__wbindgen_cb_drop=function(e){const t=a(e).original;return t.cnt--==1?(t.a=0,!0):!1},n.wbg.__wbg_Window_ec3891e998206ccf=function(e){const t=r(e).Window;return o(t)},n.wbg.__wbindgen_is_undefined=function(e){return r(e)===void 0},n.wbg.__wbg_WorkerGlobalScope_05d4962a4fb54c6a=function(e){const t=r(e).WorkerGlobalScope;return o(t)},n.wbg.__wbg_instanceof_Window_e266f02eee43b570=function(e){let t;try{t=r(e)instanceof Window}catch{t=!1}return t},n.wbg.__wbg_document_950215a728589a2d=function(e){const t=r(e).document;return P(t)?0:o(t)},n.wbg.__wbg_navigator_b18e629f7f0b75fa=function(e){const t=r(e).navigator;return o(t)},n.wbg.__wbg_querySelectorAll_608b5716e2a3baf0=function(){return w(function(e,t,_){const c=r(e).querySelectorAll(d(t,_));return o(c)},arguments)},n.wbg.__wbg_getBindGroupLayout_7478e25935b9d2e8=function(e,t){const _=r(e).getBindGroupLayout(t>>>0);return o(_)},n.wbg.__wbg_message_bf68023e199aaf1a=function(e,t){const _=r(t).message,c=m(_,f.__wbindgen_malloc,f.__wbindgen_realloc),b=g;s()[e/4+1]=b,s()[e/4+0]=c},n.wbg.__wbg_getBindGroupLayout_2eed24cc41e600f2=function(e,t){const _=r(e).getBindGroupLayout(t>>>0);return o(_)},n.wbg.__wbg_maxTextureDimension1D_7ad88ba70060cbc0=function(e){return r(e).maxTextureDimension1D},n.wbg.__wbg_maxTextureDimension2D_f618a5b67f3d6545=function(e){return r(e).maxTextureDimension2D},n.wbg.__wbg_maxTextureDimension3D_4aaaeaa186a0e6ae=function(e){return r(e).maxTextureDimension3D},n.wbg.__wbg_maxTextureArrayLayers_c40007424124dbea=function(e){return r(e).maxTextureArrayLayers},n.wbg.__wbg_maxBindGroups_82450319f50609a5=function(e){return r(e).maxBindGroups},n.wbg.__wbg_maxBindingsPerBindGroup_c1a3b5bd8ac5cb61=function(e){return r(e).maxBindingsPerBindGroup},n.wbg.__wbg_maxDynamicUniformBuffersPerPipelineLayout_e15f23e479647a3c=function(e){return r(e).maxDynamicUniformBuffersPerPipelineLayout},n.wbg.__wbg_maxDynamicStorageBuffersPerPipelineLayout_9a704b831261dd49=function(e){return r(e).maxDynamicStorageBuffersPerPipelineLayout},n.wbg.__wbg_maxSampledTexturesPerShaderStage_093bd707872fbcf6=function(e){return r(e).maxSampledTexturesPerShaderStage},n.wbg.__wbg_maxSamplersPerShaderStage_18e430a6b534706b=function(e){return r(e).maxSamplersPerShaderStage},n.wbg.__wbg_maxStorageBuffersPerShaderStage_c4d0407e3a3143a4=function(e){return r(e).maxStorageBuffersPerShaderStage},n.wbg.__wbg_maxStorageTexturesPerShaderStage_fb28b3ff3f567608=function(e){return r(e).maxStorageTexturesPerShaderStage},n.wbg.__wbg_maxUniformBuffersPerShaderStage_1243616ab1b9c3ba=function(e){return r(e).maxUniformBuffersPerShaderStage},n.wbg.__wbg_maxUniformBufferBindingSize_66afb2e3116f05a1=function(e){return r(e).maxUniformBufferBindingSize},n.wbg.__wbg_maxStorageBufferBindingSize_4c14c6ce7bff64df=function(e){return r(e).maxStorageBufferBindingSize},n.wbg.__wbg_maxVertexBuffers_0abef34c4633ebff=function(e){return r(e).maxVertexBuffers},n.wbg.__wbg_maxVertexAttributes_fae4e285196f3349=function(e){return r(e).maxVertexAttributes},n.wbg.__wbg_maxVertexBufferArrayStride_176fe097c5c78eeb=function(e){return r(e).maxVertexBufferArrayStride},n.wbg.__wbg_createView_d0df6318b34e3b5d=function(e,t){const _=r(e).createView(r(t));return o(_)},n.wbg.__wbg_destroy_95a7ca8088f60c81=function(e){r(e).destroy()},n.wbg.__wbg_setwidth_81c62bc806e0a727=function(e,t){r(e).width=t>>>0},n.wbg.__wbg_setheight_98cf0db22c40ef07=function(e,t){r(e).height=t>>>0},n.wbg.__wbg_getContext_3ae404b649cf9287=function(){return w(function(e,t,_){const c=r(e).getContext(d(t,_));return P(c)?0:o(c)},arguments)},n.wbg.__wbg_copyExternalImageToTexture_446ccb6ede6d3b9d=function(e,t,_,c){r(e).copyExternalImageToTexture(r(t),r(_),r(c))},n.wbg.__wbg_submit_145accdc4854b69b=function(e,t){r(e).submit(r(t))},n.wbg.__wbg_writeBuffer_deae9eef1958337f=function(e,t,_,c,b,u){r(e).writeBuffer(r(t),_,r(c),b,u)},n.wbg.__wbg_writeTexture_a747d2eb64753216=function(e,t,_,c,b){r(e).writeTexture(r(t),r(_),r(c),r(b))},n.wbg.__wbg_end_90bec30eeecaac54=function(e){r(e).end()},n.wbg.__wbg_executeBundles_0077022f3437c3d1=function(e,t){r(e).executeBundles(r(t))},n.wbg.__wbg_setBlendConstant_d2a884924792b10d=function(e,t){r(e).setBlendConstant(r(t))},n.wbg.__wbg_setScissorRect_0f47f59bef76ed44=function(e,t,_,c,b){r(e).setScissorRect(t>>>0,_>>>0,c>>>0,b>>>0)},n.wbg.__wbg_setStencilReference_cb3b8b016cd2622f=function(e,t){r(e).setStencilReference(t>>>0)},n.wbg.__wbg_setViewport_f78ce720ad1bbf1c=function(e,t,_,c,b,u,i){r(e).setViewport(t,_,c,b,u,i)},n.wbg.__wbg_setBindGroup_799966434e921168=function(e,t,_,c,b,u,i){r(e).setBindGroup(t>>>0,r(_),E(c,b),u,i>>>0)},n.wbg.__wbg_draw_da079c427d4e1307=function(e,t,_,c,b){r(e).draw(t>>>0,_>>>0,c>>>0,b>>>0)},n.wbg.__wbg_drawIndexed_01e94df58ffbd134=function(e,t,_,c,b,u){r(e).drawIndexed(t>>>0,_>>>0,c>>>0,b,u>>>0)},n.wbg.__wbg_drawIndexedIndirect_30c61d057fe6c676=function(e,t,_){r(e).drawIndexedIndirect(r(t),_)},n.wbg.__wbg_drawIndirect_bc41e9283103bb4c=function(e,t,_){r(e).drawIndirect(r(t),_)},n.wbg.__wbg_setIndexBuffer_90124d34472bb0a7=function(e,t,_,c){r(e).setIndexBuffer(r(t),a(_),c)},n.wbg.__wbg_setIndexBuffer_babf4a1ed7c145da=function(e,t,_,c,b){r(e).setIndexBuffer(r(t),a(_),c,b)},n.wbg.__wbg_setPipeline_4b1f6ab51617f980=function(e,t){r(e).setPipeline(r(t))},n.wbg.__wbg_setVertexBuffer_f6c24e543d847f4c=function(e,t,_,c){r(e).setVertexBuffer(t>>>0,r(_),c)},n.wbg.__wbg_setVertexBuffer_f0051e8d07a2b846=function(e,t,_,c,b){r(e).setVertexBuffer(t>>>0,r(_),c,b)},n.wbg.__wbg_instanceof_GpuCanvasContext_ed167d7e4f64d6b8=function(e){let t;try{t=r(e)instanceof GPUCanvasContext}catch{t=!1}return t},n.wbg.__wbg_configure_2eba1e396591bdd7=function(e,t){r(e).configure(r(t))},n.wbg.__wbg_getCurrentTexture_0f26ea6da8c0f77c=function(e){const t=r(e).getCurrentTexture();return o(t)},n.wbg.__wbg_error_fa1d961145d97de6=function(e){const t=r(e).error;return o(t)},n.wbg.__wbg_gpu_383beebfe7730ae8=function(e){const t=r(e).gpu;return o(t)},n.wbg.__wbg_size_6cddfc5f9d59d2be=function(e){return r(e).size},n.wbg.__wbg_usage_57ae373f36ab0f1b=function(e){return r(e).usage},n.wbg.__wbg_destroy_182829b5d1c03548=function(e){r(e).destroy()},n.wbg.__wbg_getMappedRange_33ceebd7bbe29781=function(e,t,_){const c=r(e).getMappedRange(t,_);return o(c)},n.wbg.__wbg_mapAsync_10d0f6703ef03e7b=function(e,t,_,c){const b=r(e).mapAsync(t>>>0,_,c);return o(b)},n.wbg.__wbg_unmap_ae21c65ca7ae9598=function(e){r(e).unmap()},n.wbg.__wbg_finish_bd6db27f8d9ac0ae=function(e){const t=r(e).finish();return o(t)},n.wbg.__wbg_finish_a20832e5ef22d930=function(e,t){const _=r(e).finish(r(t));return o(_)},n.wbg.__wbg_setBindGroup_964ebeee1be76825=function(e,t,_,c,b,u,i){r(e).setBindGroup(t>>>0,r(_),E(c,b),u,i>>>0)},n.wbg.__wbg_draw_7eb5cb9d384aea7b=function(e,t,_,c,b){r(e).draw(t>>>0,_>>>0,c>>>0,b>>>0)},n.wbg.__wbg_drawIndexed_81f88af371419343=function(e,t,_,c,b,u){r(e).drawIndexed(t>>>0,_>>>0,c>>>0,b,u>>>0)},n.wbg.__wbg_drawIndexedIndirect_b3596671b7a78209=function(e,t,_){r(e).drawIndexedIndirect(r(t),_)},n.wbg.__wbg_drawIndirect_199cc6179a3473bb=function(e,t,_){r(e).drawIndirect(r(t),_)},n.wbg.__wbg_setIndexBuffer_df1ee79c2996ac2d=function(e,t,_,c){r(e).setIndexBuffer(r(t),a(_),c)},n.wbg.__wbg_setIndexBuffer_7795663d5f690377=function(e,t,_,c,b){r(e).setIndexBuffer(r(t),a(_),c,b)},n.wbg.__wbg_setPipeline_4c82624264826b5e=function(e,t){r(e).setPipeline(r(t))},n.wbg.__wbg_setVertexBuffer_272d1391a3e6a158=function(e,t,_,c){r(e).setVertexBuffer(t>>>0,r(_),c)},n.wbg.__wbg_setVertexBuffer_e71e7e02b8c77fcf=function(e,t,_,c,b){r(e).setVertexBuffer(t>>>0,r(_),c,b)},n.wbg.__wbg_label_9a9e9fc564518aa6=function(e,t){const _=r(t).label,c=m(_,f.__wbindgen_malloc,f.__wbindgen_realloc),b=g;s()[e/4+1]=b,s()[e/4+0]=c},n.wbg.__wbg_beginComputePass_3a26c65b3bbaff3f=function(e,t){const _=r(e).beginComputePass(r(t));return o(_)},n.wbg.__wbg_beginRenderPass_db57aa384a7aef06=function(e,t){const _=r(e).beginRenderPass(r(t));return o(_)},n.wbg.__wbg_clearBuffer_9be070d52b051390=function(e,t,_){r(e).clearBuffer(r(t),_)},n.wbg.__wbg_clearBuffer_9c36138099ac5c3b=function(e,t,_,c){r(e).clearBuffer(r(t),_,c)},n.wbg.__wbg_copyBufferToBuffer_dfab33ec8c9e760e=function(e,t,_,c,b,u){r(e).copyBufferToBuffer(r(t),_,r(c),b,u)},n.wbg.__wbg_copyBufferToTexture_5e32ab71e42ec4c2=function(e,t,_,c){r(e).copyBufferToTexture(r(t),r(_),r(c))},n.wbg.__wbg_copyTextureToBuffer_c6674422a79a46ee=function(e,t,_,c){r(e).copyTextureToBuffer(r(t),r(_),r(c))},n.wbg.__wbg_copyTextureToTexture_bc150c40fb6fd34f=function(e,t,_,c){r(e).copyTextureToTexture(r(t),r(_),r(c))},n.wbg.__wbg_finish_72c07625138235ea=function(e){const t=r(e).finish();return o(t)},n.wbg.__wbg_finish_e43769cf456060ff=function(e,t){const _=r(e).finish(r(t));return o(_)},n.wbg.__wbg_resolveQuerySet_747a16df8fd5ab9b=function(e,t,_,c,b,u){r(e).resolveQuerySet(r(t),_>>>0,c>>>0,r(b),u>>>0)},n.wbg.__wbg_writeTimestamp_99f90a307bb33e66=function(e,t,_){r(e).writeTimestamp(r(t),_>>>0)},n.wbg.__wbg_dispatchWorkgroups_44644514248ca896=function(e,t,_,c){r(e).dispatchWorkgroups(t>>>0,_>>>0,c>>>0)},n.wbg.__wbg_dispatchWorkgroupsIndirect_74f455cf53f849df=function(e,t,_){r(e).dispatchWorkgroupsIndirect(r(t),_)},n.wbg.__wbg_end_4f73dcb320797ca5=function(e){r(e).end()},n.wbg.__wbg_setPipeline_c1c3fde5d44173c8=function(e,t){r(e).setPipeline(r(t))},n.wbg.__wbg_setBindGroup_534bbf726e58dd0d=function(e,t,_,c,b,u,i){r(e).setBindGroup(t>>>0,r(_),E(c,b),u,i>>>0)},n.wbg.__wbg_has_2b519b377040d29f=function(e,t,_){return r(e).has(d(t,_))},n.wbg.__wbg_setwidth_5f2d364182f77a59=function(e,t){r(e).width=t>>>0},n.wbg.__wbg_setheight_cc038dc5bacb3258=function(e,t){r(e).height=t>>>0},n.wbg.__wbg_gpu_b519271f1eb946a2=function(e){const t=r(e).gpu;return o(t)},n.wbg.__wbg_navigator_be23dfd02508e02f=function(e){const t=r(e).navigator;return o(t)},n.wbg.__wbg_instanceof_GpuAdapter_6a21ec3028a6a633=function(e){let t;try{t=r(e)instanceof GPUAdapter}catch{t=!1}return t},n.wbg.__wbg_features_03c1d8af712dca8d=function(e){const t=r(e).features;return o(t)},n.wbg.__wbg_limits_254f53e662b2a6f2=function(e){const t=r(e).limits;return o(t)},n.wbg.__wbg_requestDevice_98a881f5f37cbc1b=function(e,t){const _=r(e).requestDevice(r(t));return o(_)},n.wbg.__wbg_features_f50c54dfe05b4591=function(e){const t=r(e).features;return o(t)},n.wbg.__wbg_queue_6b0eedcf46d47cbd=function(e){const t=r(e).queue;return o(t)},n.wbg.__wbg_setonuncapturederror_52731b198f292e4e=function(e,t){r(e).onuncapturederror=r(t)},n.wbg.__wbg_createBindGroup_2a20ed419eb0c234=function(e,t){const _=r(e).createBindGroup(r(t));return o(_)},n.wbg.__wbg_createBindGroupLayout_d8f7eb1e6a48042e=function(e,t){const _=r(e).createBindGroupLayout(r(t));return o(_)},n.wbg.__wbg_createBuffer_7c429b6a1c19d86f=function(e,t){const _=r(e).createBuffer(r(t));return o(_)},n.wbg.__wbg_createCommandEncoder_16ef0a1822a74575=function(e,t){const _=r(e).createCommandEncoder(r(t));return o(_)},n.wbg.__wbg_createComputePipeline_9d9c4c1e7c177a43=function(e,t){const _=r(e).createComputePipeline(r(t));return o(_)},n.wbg.__wbg_createPipelineLayout_651e444b8d7b374a=function(e,t){const _=r(e).createPipelineLayout(r(t));return o(_)},n.wbg.__wbg_createQuerySet_9af179dcd7eb51f9=function(e,t){const _=r(e).createQuerySet(r(t));return o(_)},n.wbg.__wbg_createRenderBundleEncoder_a3d9e81c72356ee7=function(e,t){const _=r(e).createRenderBundleEncoder(r(t));return o(_)},n.wbg.__wbg_createRenderPipeline_adf9ebf96b9eb9b4=function(e,t){const _=r(e).createRenderPipeline(r(t));return o(_)},n.wbg.__wbg_createSampler_3246c59a5aeec1ce=function(e,t){const _=r(e).createSampler(r(t));return o(_)},n.wbg.__wbg_createShaderModule_58ad41a3299a4bb9=function(e,t){const _=r(e).createShaderModule(r(t));return o(_)},n.wbg.__wbg_createTexture_ea9e43be4047490d=function(e,t){const _=r(e).createTexture(r(t));return o(_)},n.wbg.__wbg_popErrorScope_e2a2b1b7559dad18=function(e){const t=r(e).popErrorScope();return o(t)},n.wbg.__wbg_pushErrorScope_bf4bd73394fb8138=function(e,t){r(e).pushErrorScope(a(t))},n.wbg.__wbg_instanceof_GpuOutOfMemoryError_d620da37d8112b03=function(e){let t;try{t=r(e)instanceof GPUOutOfMemoryError}catch{t=!1}return t},n.wbg.__wbg_instanceof_GpuValidationError_41d0ee6acd0ec286=function(e){let t;try{t=r(e)instanceof GPUValidationError}catch{t=!1}return t},n.wbg.__wbg_get_8187c9b59091f3ad=function(e,t){const _=r(e)[t>>>0];return P(_)?0:o(_)},n.wbg.__wbg_debug_8db2eed1bf6c1e2a=function(e){console.debug(r(e))},n.wbg.__wbg_error_fe807da27c4a4ced=function(e){console.error(r(e))},n.wbg.__wbg_info_9e6db45ac337c3b5=function(e){console.info(r(e))},n.wbg.__wbg_log_7bb108d119bafbc1=function(e){console.log(r(e))},n.wbg.__wbg_warn_e57696dbb3977030=function(e){console.warn(r(e))},n.wbg.__wbg_requestAdapter_1e9aee61dd467483=function(e,t){const _=r(e).requestAdapter(r(t));return o(_)},n.wbg.__wbg_new_b525de17f44a8943=function(){const e=new Array;return o(e)},n.wbg.__wbg_newnoargs_2b8b6bd7753c76ba=function(e,t){const _=new Function(d(e,t));return o(_)},n.wbg.__wbg_new_f841cc6f2098f4b5=function(){return o(new Map)},n.wbg.__wbg_call_95d1ea488d03e4e8=function(){return w(function(e,t){const _=r(e).call(r(t));return o(_)},arguments)},n.wbg.__wbg_new_f9876326328f45ed=function(){const e=new Object;return o(e)},n.wbg.__wbg_self_e7c1f827057f6584=function(){return w(function(){const e=self.self;return o(e)},arguments)},n.wbg.__wbg_window_a09ec664e14b1b81=function(){return w(function(){const e=window.window;return o(e)},arguments)},n.wbg.__wbg_globalThis_87cbb8506fecf3a9=function(){return w(function(){const e=globalThis.globalThis;return o(e)},arguments)},n.wbg.__wbg_global_c85a9259e621f3db=function(){return w(function(){const e=global.global;return o(e)},arguments)},n.wbg.__wbg_set_17224bc548dd1d7b=function(e,t,_){r(e)[t>>>0]=a(_)},n.wbg.__wbg_push_49c286f04dd3bf59=function(e,t){return r(e).push(r(t))},n.wbg.__wbg_call_9495de66fdbe016b=function(){return w(function(e,t,_){const c=r(e).call(r(t),r(_));return o(c)},arguments)},n.wbg.__wbg_set_388c4c6422704173=function(e,t,_){const c=r(e).set(r(t),r(_));return o(c)},n.wbg.__wbg_instanceof_Object_f5a826c4da0d4a94=function(e){let t;try{t=r(e)instanceof Object}catch{t=!1}return t},n.wbg.__wbg_valueOf_1e54bbd68df19aa2=function(e){const t=r(e).valueOf();return o(t)},n.wbg.__wbg_new_9d3a9ce4282a18a8=function(e,t){try{var _={a:e,b:t},c=(u,i)=>{const p=_.a;_.a=0;try{return q(p,_.b,u,i)}finally{_.a=p}};const b=new Promise(c);return o(b)}finally{_.a=_.b=0}},n.wbg.__wbg_resolve_fd40f858d9db1a04=function(e){const t=Promise.resolve(r(e));return o(t)},n.wbg.__wbg_then_ec5db6d509eb475f=function(e,t){const _=r(e).then(r(t));return o(_)},n.wbg.__wbg_then_f753623316e2873a=function(e,t,_){const c=r(e).then(r(t),r(_));return o(c)},n.wbg.__wbg_buffer_cf65c07de34b9a08=function(e){const t=r(e).buffer;return o(t)},n.wbg.__wbg_newwithbyteoffsetandlength_9fb2f11355ecadf5=function(e,t,_){const c=new Uint8Array(r(e),t>>>0,_>>>0);return o(c)},n.wbg.__wbg_new_537b7341ce90bb31=function(e){const t=new Uint8Array(r(e));return o(t)},n.wbg.__wbg_set_17499e8aa4003ebd=function(e,t,_){r(e).set(r(t),_>>>0)},n.wbg.__wbg_length_27a2afe8ab42b09f=function(e){return r(e).length},n.wbg.__wbg_buffer_5f1fc856188c4b44=function(e){const t=r(e).buffer;return o(t)},n.wbg.__wbg_set_6aa458a4ebdb65cb=function(){return w(function(e,t,_){return Reflect.set(r(e),r(t),r(_))},arguments)},n.wbg.__wbindgen_debug_string=function(e,t){const _=O(r(t)),c=m(_,f.__wbindgen_malloc,f.__wbindgen_realloc),b=g;s()[e/4+1]=b,s()[e/4+0]=c},n.wbg.__wbindgen_throw=function(e,t){throw new Error(d(e,t))},n.wbg.__wbindgen_memory=function(){const e=f.memory;return o(e)},n.wbg.__wbindgen_closure_wrapper1231=function(e,t,_){const c=G(e,t,411,C);return o(c)},n.wbg.__wbindgen_closure_wrapper1233=function(e,t,_){const c=G(e,t,411,C);return o(c)},n.wbg.__wbindgen_closure_wrapper3785=function(e,t,_){const c=G(e,t,881,U);return o(c)},n}function N(n,e){return f=n.exports,Q.__wbindgen_wasm_module=e,x=null,h=null,B=null,y=null,f.__wbindgen_start(),f}async function Q(n){typeof n=="undefined"&&(n=new URL("/chat_wasm/assets/wonnx_bg.95cd39d5.wasm",self.location));const e=z();(typeof n=="string"||typeof Request=="function"&&n instanceof Request||typeof URL=="function"&&n instanceof URL)&&(n=fetch(n));const{instance:t,module:_}=await $(await n,e);return N(t,_)}export{I,W as S,Q as i};
