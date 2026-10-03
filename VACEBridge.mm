#include "VACEBridge.h"
#include "stable-diffusion.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct VACEHandle { sd_ctx_t *ctx=nullptr; };
static thread_local std::string g_error;
static void log_cb(enum sd_log_level_t, const char *text, void*) { if(text) g_error=text; }
VACEHandle *vace_create(const char *diffusion,const char *vae,const char *t5){
  g_error.clear(); if(!diffusion||!vae||!t5){g_error="Missing VACE model paths";return nullptr;}
  sd_ctx_params_t p; sd_ctx_params_init(&p); p.diffusion_model_path=diffusion; p.vae_path=vae; p.t5xxl_path=t5; p.enable_mmap=true; p.flash_attn=true; p.diffusion_flash_attn=true; p.diffusion_conv_direct=true; p.vae_conv_direct=true; p.n_threads=4; p.backend="metal"; p.params_backend="cpu";
  sd_set_log_callback(log_cb,nullptr); auto *h=(VACEHandle*)calloc(1,sizeof(VACEHandle)); h->ctx=new_sd_ctx(&p); if(!h->ctx){free(h); if(g_error.empty())g_error="Could not load VACE model"; return nullptr;} return h;
}
void vace_destroy(VACEHandle*h){if(!h)return; if(h->ctx)free_sd_ctx(h->ctx); free(h);}
void vace_cancel(VACEHandle*h){if(h&&h->ctx)sd_cancel_generation(h->ctx,SD_CANCEL_ALL);}
const char*vace_last_error(void){return g_error.c_str();}
int vace_generate(VACEHandle*h,const uint8_t*rgba,int frame_count,int width,int height,int fps,const char*prompt,const char*negative_prompt,int steps,float strength,int64_t seed,uint8_t**out_rgba,int*out_count,int*out_fps){
 g_error.clear(); if(!h||!h->ctx||!rgba||frame_count<5||width<32||height<32||!out_rgba||!out_count){g_error="Invalid VACE generation request";return 0;}
 std::vector<sd_image_t> frames(frame_count); for(int i=0;i<frame_count;i++){frames[i]={((uint32_t)width),((uint32_t)height),4,const_cast<uint8_t*>(rgba+(size_t)i*width*height*4)};}
 sd_vid_gen_params_t p; sd_vid_gen_params_init(&p); p.prompt=prompt?prompt:""; p.negative_prompt=negative_prompt?negative_prompt:""; p.width=width; p.height=height; p.fps=fps>0?fps:16; p.video_frames=frame_count; p.seed=seed; p.strength=strength; p.vace_strength=1.0f; p.sample_params.sample_steps=steps>0?steps:8; p.sample_params.guidance.txt_cfg=5.0f; p.control_frames=frames.data(); p.control_frames_size=frame_count;
 sd_image_t *out=nullptr; int count=0,ofps=0; if(!generate_video(h->ctx,&p,&out,&count,nullptr,&ofps)||!out||count<=0){if(g_error.empty())g_error="VACE generation failed"; if(out)free_sd_images(out,count);return 0;}
 size_t bytes=(size_t)count*width*height*4; uint8_t*copy=(uint8_t*)malloc(bytes); if(!copy){free_sd_images(out,count);g_error="Out of memory for VACE output";return 0;}
 for(int i=0;i<count;i++){if(out[i].channel==4)memcpy(copy+(size_t)i*width*height*4,out[i].data,width*height*4);else{for(size_t j=0;j<(size_t)width*height;j++){auto*src=out[i].data+j*out[i].channel;auto*dst=copy+(size_t)i*width*height*4+j*4;dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];dst[3]=255;}}}
 free_sd_images(out,count); *out_rgba=copy; *out_count=count; if(out_fps)*out_fps=ofps>0?ofps:p.fps; return 1;
}
void vace_free_frames(uint8_t*f){free(f);}
