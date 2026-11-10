#include "X265PresetTranscode.h"
#include "X265PresetParams.h"

#include "x265.h"

#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static char g_param_error[512];

static void log_callback(void *avcl, int level, const char *fmt, va_list args) {
    char line[512];
    va_list copy;
    va_copy(copy, args);
    vsnprintf(line, sizeof(line), fmt, copy);
    va_end(copy);
    if ((strstr(line, "Unknown option") || strstr(line, "Invalid value")) && g_param_error[0] == '\0') {
        memcpy(g_param_error, line, sizeof(g_param_error));
    }
    if (level <= AV_LOG_ERROR) {
        va_list printed;
        va_copy(printed, args);
        av_log_default_callback(avcl, level, fmt, printed);
        va_end(printed);
    }
}

static void set_error(char *error_out, size_t error_out_size, const char *fmt, ...) {
    if (!error_out || error_out_size == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(error_out, error_out_size, fmt, args);
    va_end(args);
}

static int fail(char *error_out, size_t error_out_size, int err, const char *what) {
    char detail[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, detail, sizeof(detail));
    set_error(error_out, error_out_size, "%s: %s", what, detail);
    return -1;
}

static int cancelled(volatile int32_t *cancel_flag) {
    return cancel_flag && *cancel_flag;
}

static int interrupt_cb(void *opaque) {
    return cancelled(opaque);
}

static void rotation_prefix(const AVStream *stream, char *out, size_t out_size) {
    out[0] = '\0';
    const AVPacketSideData *side = av_packet_side_data_get(
        stream->codecpar->coded_side_data,
        stream->codecpar->nb_coded_side_data,
        AV_PKT_DATA_DISPLAYMATRIX);
    if (!side || side->size < 9 * (int)sizeof(int32_t)) return;
    double theta = -av_display_rotation_get((const int32_t *)side->data);
    if (isnan(theta)) return;
    while (theta < 0) theta += 360.0;
    while (theta >= 360.0) theta -= 360.0;
    if (fabs(theta - 90.0) < 1.0) snprintf(out, out_size, "transpose=clock,");
    else if (fabs(theta - 270.0) < 1.0) snprintf(out, out_size, "transpose=cclock,");
    else if (fabs(theta - 180.0) < 1.0) snprintf(out, out_size, "hflip,vflip,");
}

static int64_t expected_output_frames(const AVStream *stream) {
    if (!stream || stream->duration <= 0 || stream->time_base.den <= 0) return 0;
    const double seconds = (double)stream->duration * stream->time_base.num / stream->time_base.den;
    if (!(seconds > 0)) return 0;
    return (int64_t)(seconds * 24000.0 / 1001.0 + 0.5);
}

static int open_decoder(AVCodecContext **context, const AVStream *stream, char *error_out, size_t error_out_size) {
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        set_error(error_out, error_out_size, "No decoder for stream %d.", stream->index);
        return -1;
    }
    AVCodecContext *decoder = avcodec_alloc_context3(codec);
    if (!decoder) {
        set_error(error_out, error_out_size, "Could not allocate a decoder.");
        return -1;
    }
    int err = avcodec_parameters_to_context(decoder, stream->codecpar);
    if (err < 0) {
        avcodec_free_context(&decoder);
        return fail(error_out, error_out_size, err, "Could not copy decoder parameters");
    }
    decoder->pkt_timebase = stream->time_base;
    decoder->thread_count = 1;
    err = avcodec_open2(decoder, codec, NULL);
    if (err < 0) {
        avcodec_free_context(&decoder);
        return fail(error_out, error_out_size, err, "Could not open the decoder");
    }
    *context = decoder;
    return 0;
}

static int open_video_encoder(AVCodecContext **context, char *error_out, size_t error_out_size) {
    const AVCodec *codec = avcodec_find_encoder_by_name("libx265");
    if (!codec) {
        set_error(error_out, error_out_size, "This build does not include libx265.");
        return -1;
    }
    AVCodecContext *encoder = avcodec_alloc_context3(codec);
    if (!encoder) {
        set_error(error_out, error_out_size, "Could not allocate the x265 encoder.");
        return -1;
    }
    encoder->width = 1920;
    encoder->height = 1080;
    encoder->pix_fmt = AV_PIX_FMT_YUV420P10LE;
    encoder->time_base = (AVRational){1001, 24000};
    encoder->framerate = (AVRational){24000, 1001};
    encoder->color_range = AVCOL_RANGE_MPEG;
    encoder->color_primaries = AVCOL_PRI_UNSPECIFIED;
    encoder->color_trc = AVCOL_TRC_UNSPECIFIED;
    encoder->colorspace = AVCOL_SPC_UNSPECIFIED;
    encoder->chroma_sample_location = AVCHROMA_LOC_UNSPECIFIED;
    encoder->sample_aspect_ratio = (AVRational){0, 1};
    encoder->gop_size = 250;
    encoder->keyint_min = 23;
    encoder->max_b_frames = 4;
    encoder->thread_count = 0;
    encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    g_param_error[0] = '\0';
    if (av_opt_set(encoder->priv_data, "preset", "medium", 0) < 0 ||
        av_opt_set(encoder->priv_data, "profile", "main10", 0) < 0 ||
        av_opt_set(encoder->priv_data, "x265-params", RIFE60_X265_PRESET_PARAMS, 0) < 0) {
        avcodec_free_context(&encoder);
        set_error(error_out, error_out_size, "Could not apply the x265 preset.");
        return -1;
    }
    int err = avcodec_open2(encoder, codec, NULL);
    if (g_param_error[0] != '\0') {
        avcodec_free_context(&encoder);
        set_error(error_out, error_out_size, "x265 rejected a preset option: %s", g_param_error);
        return -1;
    }
    if (err < 0) {
        avcodec_free_context(&encoder);
        return fail(error_out, error_out_size, err, "Could not open libx265");
    }
    *context = encoder;
    return 0;
}

static int open_audio_encoder(AVCodecContext **context, char *error_out, size_t error_out_size) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec) {
        set_error(error_out, error_out_size, "This build does not include the AAC encoder.");
        return -1;
    }
    AVCodecContext *encoder = avcodec_alloc_context3(codec);
    if (!encoder) {
        set_error(error_out, error_out_size, "Could not allocate the AAC encoder.");
        return -1;
    }
    encoder->bit_rate = 256000;
    encoder->sample_rate = 48000;
    encoder->sample_fmt = AV_SAMPLE_FMT_FLTP;
    encoder->profile = FF_PROFILE_AAC_LOW;
    encoder->time_base = (AVRational){1, 48000};
    av_channel_layout_default(&encoder->ch_layout, 2);
    int err = avcodec_open2(encoder, codec, NULL);
    if (err < 0) {
        avcodec_free_context(&encoder);
        return fail(error_out, error_out_size, err, "Could not open the AAC encoder");
    }
    *context = encoder;
    return 0;
}

static int configure_filter(AVFilterGraph **graph_out, AVFilterContext **src_out, AVFilterContext **sink_out,
                            const AVFrame *frame, const char *rotation, char *error_out, size_t error_out_size) {
    AVFilterGraph *graph = avfilter_graph_alloc();
    const AVFilter *buffer = avfilter_get_by_name("buffer");
    const AVFilter *sink = avfilter_get_by_name("buffersink");
    if (!graph || !buffer || !sink) {
        avfilter_graph_free(&graph);
        set_error(error_out, error_out_size, "The scale/fps filters are not available in this FFmpeg build.");
        return -1;
    }
    AVRational time_base = frame->time_base.num > 0 && frame->time_base.den > 0 ? frame->time_base : (AVRational){1, 24000};
    AVRational sar = frame->sample_aspect_ratio.num > 0 ? frame->sample_aspect_ratio : (AVRational){1, 1};
    char args[256];
    snprintf(args, sizeof(args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
             frame->width, frame->height, frame->format, time_base.num, time_base.den, sar.num, sar.den);
    AVFilterContext *source = NULL;
    AVFilterContext *destination = NULL;
    int err = avfilter_graph_create_filter(&source, buffer, "in", args, NULL, graph);
    if (err < 0) {
        avfilter_graph_free(&graph);
        return fail(error_out, error_out_size, err, "Could not create the video buffer");
    }
    err = avfilter_graph_create_filter(&destination, sink, "out", NULL, NULL, graph);
    if (err < 0) {
        avfilter_graph_free(&graph);
        return fail(error_out, error_out_size, err, "Could not create the video sink");
    }
    enum AVPixelFormat formats[] = {AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_NONE};
    err = av_opt_set_int_list(destination, "pix_fmts", formats, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (err < 0) {
        avfilter_graph_free(&graph);
        return fail(error_out, error_out_size, err, "Could not request yuv420p10le");
    }
    char description[512];
    snprintf(description, sizeof(description),
             "%sscale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease,"
             "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=24000/1001,format=yuv420p10le",
             rotation ? rotation : "");
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs = avfilter_inout_alloc();
    if (!outputs || !inputs) {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        avfilter_graph_free(&graph);
        set_error(error_out, error_out_size, "Could not allocate the filter graph.");
        return -1;
    }
    outputs->name = av_strdup("in");
    outputs->filter_ctx = source;
    outputs->pad_idx = 0;
    outputs->next = NULL;
    inputs->name = av_strdup("out");
    inputs->filter_ctx = destination;
    inputs->pad_idx = 0;
    inputs->next = NULL;
    err = avfilter_graph_parse_ptr(graph, description, &inputs, &outputs, NULL);
    if (err >= 0) err = avfilter_graph_config(graph, NULL);
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    if (err < 0) {
        avfilter_graph_free(&graph);
        return fail(error_out, error_out_size, err, "Could not configure scale and frame rate");
    }
    *graph_out = graph;
    *src_out = source;
    *sink_out = destination;
    return 0;
}

static int write_encoded(AVFormatContext *output, AVCodecContext *encoder, AVStream *stream, AVPacket *packet) {
    av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
    packet->stream_index = stream->index;
    int err = av_interleaved_write_frame(output, packet);
    av_packet_unref(packet);
    return err;
}

static int drain_video(AVFormatContext *output, AVCodecContext *encoder, AVStream *stream,
                       AVPacket *packet, int64_t *frames, int64_t expected,
                       int (*progress)(int64_t, int64_t, void *), void *userdata,
                       volatile int32_t *cancel_flag, char *error_out, size_t error_out_size) {
    while (1) {
        if (cancelled(cancel_flag)) return 1;
        int err = avcodec_receive_packet(encoder, packet);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return 0;
        if (err < 0) return fail(error_out, error_out_size, err, "x265 output failed");
        err = write_encoded(output, encoder, stream, packet);
        if (err < 0) return fail(error_out, error_out_size, err, "Could not write the x265 video");
        *frames += 1;
        if (progress && progress(*frames, expected, userdata)) return 1;
    }
}

static int send_video_frame(AVFormatContext *output, AVCodecContext *encoder, AVStream *stream,
                            AVFrame *frame, AVPacket *packet, int64_t *frames, int64_t expected,
                            int (*progress)(int64_t, int64_t, void *), void *userdata,
                            volatile int32_t *cancel_flag, char *error_out, size_t error_out_size) {
    int err = avcodec_send_frame(encoder, frame);
    if (err < 0) return fail(error_out, error_out_size, err, "x265 rejected a frame");
    return drain_video(output, encoder, stream, packet, frames, expected, progress, userdata, cancel_flag, error_out, error_out_size);
}

static int pull_filtered(AVFilterContext *sink, AVFormatContext *output, AVCodecContext *encoder, AVStream *stream,
                         AVFrame *filtered, AVPacket *packet, int64_t *frames, int64_t *scheduled, int64_t expected,
                         int (*progress)(int64_t, int64_t, void *), void *userdata,
                         volatile int32_t *cancel_flag, char *error_out, size_t error_out_size) {
    while (1) {
        if (cancelled(cancel_flag)) return 1;
        int err = av_buffersink_get_frame(sink, filtered);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return 0;
        if (err < 0) return fail(error_out, error_out_size, err, "Scale or frame-rate conversion failed");
        filtered->pts = (*scheduled)++;
        if (progress && progress(*scheduled, expected, userdata)) {
            av_frame_unref(filtered);
            return 1;
        }
        err = send_video_frame(output, encoder, stream, filtered, packet, frames, expected, NULL, userdata, cancel_flag, error_out, error_out_size);
        av_frame_unref(filtered);
        if (err != 0) return err;
    }
}

static int encode_audio_frame(AVFormatContext *output, AVCodecContext *encoder, AVStream *stream,
                              AVFrame *frame, AVPacket *packet, char *error_out, size_t error_out_size) {
    int err = avcodec_send_frame(encoder, frame);
    if (err < 0) return fail(error_out, error_out_size, err, "AAC rejected audio");
    while (1) {
        err = avcodec_receive_packet(encoder, packet);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return 0;
        if (err < 0) return fail(error_out, error_out_size, err, "AAC output failed");
        err = write_encoded(output, encoder, stream, packet);
        if (err < 0) return fail(error_out, error_out_size, err, "Could not write AAC audio");
    }
}

static int drain_fifo(AVAudioFifo *fifo, AVCodecContext *encoder, AVFormatContext *output, AVStream *stream,
                      AVFrame *frame, AVPacket *packet, int64_t *samples, int final,
                      char *error_out, size_t error_out_size) {
    const int frame_size = encoder->frame_size > 0 ? encoder->frame_size : 1024;
    while (av_audio_fifo_size(fifo) >= frame_size || (final && av_audio_fifo_size(fifo) > 0)) {
        const int count = final && av_audio_fifo_size(fifo) < frame_size ? av_audio_fifo_size(fifo) : frame_size;
        frame->nb_samples = count;
        frame->format = encoder->sample_fmt;
        frame->sample_rate = encoder->sample_rate;
        if (av_channel_layout_copy(&frame->ch_layout, &encoder->ch_layout) < 0) {
            set_error(error_out, error_out_size, "Could not describe the AAC channel layout.");
            return -1;
        }
        int err = av_frame_get_buffer(frame, 0);
        if (err < 0) return fail(error_out, error_out_size, err, "Could not allocate an AAC frame");
        if (av_audio_fifo_read(fifo, (void **)frame->data, count) < count) {
            av_frame_unref(frame);
            set_error(error_out, error_out_size, "The audio resampler ran out of samples.");
            return -1;
        }
        frame->pts = *samples;
        *samples += count;
        err = encode_audio_frame(output, encoder, stream, frame, packet, error_out, error_out_size);
        av_frame_unref(frame);
        if (err != 0) return err;
        if (final && av_audio_fifo_size(fifo) <= 0) break;
    }
    return 0;
}

__attribute__((visibility("default"), used))
const char *rife60_x265_preset_params(void) {
    return RIFE60_X265_PRESET_PARAMS;
}

__attribute__((visibility("default"), used))
int rife60_x265_build_number(void) {
    const x265_api *api = x265_api_get(10);
    return api ? api->api_build_number : 0;
}

__attribute__((visibility("default"), used))
int rife60_x265_preset_transcode(
    const char *input_path,
    const char *output_path,
    volatile int32_t *cancel_flag,
    int (*progress)(int64_t frame, int64_t total, void *userdata),
    void *userdata,
    char *error_out,
    size_t error_out_size) {
    if (error_out && error_out_size) error_out[0] = '\0';
    av_log_set_callback(log_callback);
    av_log_set_level(AV_LOG_ERROR);

    AVFormatContext *input = avformat_alloc_context();
    AVFormatContext *output = NULL;
    AVCodecContext *video_decoder = NULL;
    AVCodecContext *audio_decoder = NULL;
    AVCodecContext *video_encoder = NULL;
    AVCodecContext *audio_encoder = NULL;
    AVFilterGraph *graph = NULL;
    AVFilterContext *source = NULL;
    AVFilterContext *sink = NULL;
    SwrContext *resampler = NULL;
    AVAudioFifo *fifo = NULL;
    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded = av_frame_alloc();
    AVFrame *filtered = av_frame_alloc();
    AVFrame *audio = av_frame_alloc();
    AVStream *video_out = NULL;
    AVStream *audio_out = NULL;
    int video_index = -1;
    int audio_index = -1;
    int64_t video_frames = 0;
    int64_t scheduled = 0;
    int64_t expected = 0;
    int64_t audio_pts = 0;
    int video_packets = 0;
    int decoded_frames = 0;
    int result = -1;
    char rotation[64];

    if (!input || !packet || !decoded || !filtered || !audio) {
        set_error(error_out, error_out_size, "Could not allocate the x265 transcode.");
        goto done;
    }
    input->interrupt_callback.callback = interrupt_cb;
    input->interrupt_callback.opaque = (void *)cancel_flag;
    int err = avformat_open_input(&input, input_path, NULL, NULL);
    if (err < 0) {
        result = fail(error_out, error_out_size, err, "Could not open the source");
        goto done;
    }
    err = avformat_find_stream_info(input, NULL);
    if (err < 0) {
        result = fail(error_out, error_out_size, err, "Could not read the source streams");
        goto done;
    }
    for (unsigned i = 0; i < input->nb_streams; i++) {
        if (video_index < 0 && input->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            video_index = (int)i;
        else if (audio_index < 0 && input->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
            audio_index = (int)i;
    }
    if (video_index < 0) {
        set_error(error_out, error_out_size, "The source has no video track.");
        goto done;
    }
    if (open_decoder(&video_decoder, input->streams[video_index], error_out, error_out_size) != 0) goto done;
    if (audio_index >= 0 && open_decoder(&audio_decoder, input->streams[audio_index], error_out, error_out_size) != 0) goto done;
    if (open_video_encoder(&video_encoder, error_out, error_out_size) != 0) goto done;
    if (audio_decoder && open_audio_encoder(&audio_encoder, error_out, error_out_size) != 0) goto done;

    err = avformat_alloc_output_context2(&output, NULL, "mov", output_path);
    if (err < 0 || !output) {
        result = fail(error_out, error_out_size, err < 0 ? err : AVERROR_UNKNOWN, "Could not create the MOV output");
        goto done;
    }
    video_out = avformat_new_stream(output, NULL);
    if (!video_out) {
        set_error(error_out, error_out_size, "Could not add the output video track.");
        goto done;
    }
    err = avcodec_parameters_from_context(video_out->codecpar, video_encoder);
    if (err < 0) {
        result = fail(error_out, error_out_size, err, "Could not describe the x265 track");
        goto done;
    }
    video_out->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
    video_out->time_base = video_encoder->time_base;
    video_out->avg_frame_rate = video_encoder->framerate;
    if (audio_encoder) {
        audio_out = avformat_new_stream(output, NULL);
        if (!audio_out) {
            set_error(error_out, error_out_size, "Could not add the output audio track.");
            goto done;
        }
        err = avcodec_parameters_from_context(audio_out->codecpar, audio_encoder);
        if (err < 0) {
            result = fail(error_out, error_out_size, err, "Could not describe the AAC track");
            goto done;
        }
        audio_out->time_base = audio_encoder->time_base;
        AVChannelLayout stereo;
        av_channel_layout_default(&stereo, 2);
        resampler = NULL;
        err = swr_alloc_set_opts2(&resampler, &audio_encoder->ch_layout, audio_encoder->sample_fmt, audio_encoder->sample_rate,
                                  &audio_decoder->ch_layout, audio_decoder->sample_fmt, audio_decoder->sample_rate, 0, NULL);
        av_channel_layout_uninit(&stereo);
        if (err < 0 || !resampler || swr_init(resampler) < 0) {
            set_error(error_out, error_out_size, "Could not set up AAC stereo 48 kHz conversion.");
            goto done;
        }
        fifo = av_audio_fifo_alloc(audio_encoder->sample_fmt, 2, audio_encoder->frame_size > 0 ? audio_encoder->frame_size * 4 : 4096);
        if (!fifo) {
            set_error(error_out, error_out_size, "Could not allocate the audio buffer.");
            goto done;
        }
    }
    if (!(output->oformat->flags & AVFMT_NOFILE)) {
        err = avio_open(&output->pb, output_path, AVIO_FLAG_WRITE);
        if (err < 0) {
            result = fail(error_out, error_out_size, err, "Could not open the output file");
            goto done;
        }
    }
    AVDictionary *mux_options = NULL;
    av_dict_set(&mux_options, "movflags", "+faststart", 0);
    err = avformat_write_header(output, &mux_options);
    av_dict_free(&mux_options);
    if (err < 0) {
        result = fail(error_out, error_out_size, err, "Could not start the MOV file");
        goto done;
    }
    if (video_out->codecpar->codec_tag != MKTAG('h', 'v', 'c', '1')) {
        set_error(error_out, error_out_size, "The MOV muxer did not keep the hvc1 tag.");
        goto done;
    }

    rotation_prefix(input->streams[video_index], rotation, sizeof(rotation));
    expected = expected_output_frames(input->streams[video_index]);
    if (progress && progress(0, expected, userdata)) {
        result = 1;
        goto done;
    }

    while (1) {
        if (cancelled(cancel_flag)) { result = 1; goto done; }
        err = av_read_frame(input, packet);
        if (err == AVERROR_EOF) break;
        if (err < 0) {
            result = fail(error_out, error_out_size, err, "Could not read the source");
            goto done;
        }
        if (packet->stream_index == video_index) {
            video_packets += 1;
            err = avcodec_send_packet(video_decoder, packet);
            av_packet_unref(packet);
            if (err < 0) {
                result = fail(error_out, error_out_size, err, "Video decode failed");
                goto done;
            }
            while ((err = avcodec_receive_frame(video_decoder, decoded)) >= 0) {
                decoded_frames += 1;
                decoded->time_base = input->streams[video_index]->time_base;
                if (!graph) {
                    if (configure_filter(&graph, &source, &sink, decoded, rotation, error_out, error_out_size) != 0) {
                        result = -1;
                        goto done;
                    }
                }
                err = av_buffersrc_add_frame(source, decoded);
                av_frame_unref(decoded);
                if (err < 0) {
                    result = fail(error_out, error_out_size, err, "Could not scale a frame");
                    goto done;
                }
                int step = pull_filtered(sink, output, video_encoder, video_out, filtered, packet, &video_frames, &scheduled, expected,
                                         progress, userdata, cancel_flag, error_out, error_out_size);
                if (step != 0) { result = step; goto done; }
            }
            if (err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                result = fail(error_out, error_out_size, err, "Video decode failed");
                goto done;
            }
        } else if (audio_decoder && packet->stream_index == audio_index) {
            err = avcodec_send_packet(audio_decoder, packet);
            av_packet_unref(packet);
            if (err < 0) {
                result = fail(error_out, error_out_size, err, "Audio decode failed");
                goto done;
            }
            while ((err = avcodec_receive_frame(audio_decoder, decoded)) >= 0) {
                uint8_t **converted = NULL;
                int out_count = (int)av_rescale_rnd(swr_get_delay(resampler, decoded->sample_rate) + decoded->nb_samples,
                                                     audio_encoder->sample_rate, decoded->sample_rate, AV_ROUND_UP);
                if (av_samples_alloc_array_and_samples(&converted, NULL, 2, out_count, audio_encoder->sample_fmt, 0) < 0) {
                    av_frame_unref(decoded);
                    set_error(error_out, error_out_size, "Could not allocate converted audio.");
                    goto done;
                }
                int converted_count = swr_convert(resampler, converted, out_count,
                                                  (const uint8_t **)decoded->extended_data, decoded->nb_samples);
                av_frame_unref(decoded);
                if (converted_count < 0) {
                    av_freep(&converted[0]);
                    av_freep(&converted);
                    result = fail(error_out, error_out_size, converted_count, "Audio conversion failed");
                    goto done;
                }
                if (converted_count > 0 && av_audio_fifo_write(fifo, (void **)converted, converted_count) < converted_count) {
                    av_freep(&converted[0]);
                    av_freep(&converted);
                    set_error(error_out, error_out_size, "Could not buffer converted audio.");
                    goto done;
                }
                av_freep(&converted[0]);
                av_freep(&converted);
                int step = drain_fifo(fifo, audio_encoder, output, audio_out, audio, packet, &audio_pts, 0, error_out, error_out_size);
                if (step != 0) { result = step; goto done; }
            }
            if (err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                result = fail(error_out, error_out_size, err, "Audio decode failed");
                goto done;
            }
        } else {
            av_packet_unref(packet);
        }
    }

    if (video_decoder) {
        avcodec_send_packet(video_decoder, NULL);
        while ((err = avcodec_receive_frame(video_decoder, decoded)) >= 0) {
            decoded_frames += 1;
            decoded->time_base = input->streams[video_index]->time_base;
            if (!graph && configure_filter(&graph, &source, &sink, decoded, rotation, error_out, error_out_size) != 0) {
                result = -1;
                goto done;
            }
            err = av_buffersrc_add_frame(source, decoded);
            av_frame_unref(decoded);
            if (err < 0) {
                result = fail(error_out, error_out_size, err, "Could not scale a frame");
                goto done;
            }
            int step = pull_filtered(sink, output, video_encoder, video_out, filtered, packet, &video_frames, &scheduled, expected,
                                     progress, userdata, cancel_flag, error_out, error_out_size);
            if (step != 0) { result = step; goto done; }
        }
    }
    if (source) {
        err = av_buffersrc_add_frame(source, NULL);
        if (err < 0) {
            result = fail(error_out, error_out_size, err, "Could not finish scaling");
            goto done;
        }
        int step = pull_filtered(sink, output, video_encoder, video_out, filtered, packet, &video_frames, &scheduled, expected,
                                 progress, userdata, cancel_flag, error_out, error_out_size);
        if (step != 0) { result = step; goto done; }
    }
    if (!graph || scheduled <= 0) {
        set_error(error_out, error_out_size,
                  "The source produced no video frames (packets=%d decoded=%d).",
                  video_packets, decoded_frames);
        result = -1;
        goto done;
    }
    int flushed = send_video_frame(output, video_encoder, video_out, NULL, packet, &video_frames, expected,
                                   progress, userdata, cancel_flag, error_out, error_out_size);
    if (flushed != 0) { result = flushed; goto done; }

    if (audio_decoder && fifo) {
        avcodec_send_packet(audio_decoder, NULL);
        while (avcodec_receive_frame(audio_decoder, decoded) >= 0) {
            uint8_t **converted = NULL;
            int out_count = (int)av_rescale_rnd(swr_get_delay(resampler, decoded->sample_rate) + decoded->nb_samples,
                                                 audio_encoder->sample_rate, decoded->sample_rate, AV_ROUND_UP);
            if (av_samples_alloc_array_and_samples(&converted, NULL, 2, out_count > 0 ? out_count : 1, audio_encoder->sample_fmt, 0) < 0) {
                av_frame_unref(decoded);
                set_error(error_out, error_out_size, "Could not allocate converted audio.");
                goto done;
            }
            int converted_count = swr_convert(resampler, converted, out_count, (const uint8_t **)decoded->extended_data, decoded->nb_samples);
            av_frame_unref(decoded);
            if (converted_count > 0) av_audio_fifo_write(fifo, (void **)converted, converted_count);
            av_freep(&converted[0]);
            av_freep(&converted);
        }
        uint8_t **converted = NULL;
        int delayed = (int)swr_get_delay(resampler, audio_encoder->sample_rate);
        if (delayed > 0 && av_samples_alloc_array_and_samples(&converted, NULL, 2, delayed, audio_encoder->sample_fmt, 0) >= 0) {
            int converted_count = swr_convert(resampler, converted, delayed, NULL, 0);
            if (converted_count > 0) av_audio_fifo_write(fifo, (void **)converted, converted_count);
            av_freep(&converted[0]);
            av_freep(&converted);
        }
        int step = drain_fifo(fifo, audio_encoder, output, audio_out, audio, packet, &audio_pts, 1, error_out, error_out_size);
        if (step != 0) { result = step; goto done; }
        step = encode_audio_frame(output, audio_encoder, audio_out, NULL, packet, error_out, error_out_size);
        if (step != 0) { result = step; goto done; }
    }

    err = av_write_trailer(output);
    if (err < 0) {
        result = fail(error_out, error_out_size, err, "Could not finish the MOV file");
        goto done;
    }
    if (progress) progress(video_frames, video_frames, userdata);
    result = 0;

done:
    av_frame_free(&audio);
    av_frame_free(&filtered);
    av_frame_free(&decoded);
    av_packet_free(&packet);
    if (fifo) av_audio_fifo_free(fifo);
    swr_free(&resampler);
    avfilter_graph_free(&graph);
    avcodec_free_context(&audio_encoder);
    avcodec_free_context(&video_encoder);
    avcodec_free_context(&audio_decoder);
    avcodec_free_context(&video_decoder);
    if (output) {
        if (!(output->oformat->flags & AVFMT_NOFILE) && output->pb) avio_closep(&output->pb);
        avformat_free_context(output);
    }
    avformat_close_input(&input);
    av_log_set_callback(av_log_default_callback);
    return result;
}

#ifdef RIFE60_X265_PRESET_TOOL
static int tool_progress(int64_t frame, int64_t total, void *userdata) {
    (void)userdata;
    fprintf(stderr, "frame %lld / %lld\n", (long long)frame, (long long)total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: x265-preset-transcode input output.mov\n");
        return 2;
    }
    char error[512] = {0};
    int rc = rife60_x265_preset_transcode(argv[1], argv[2], NULL, tool_progress, NULL, error, sizeof(error));
    if (rc != 0) {
        fprintf(stderr, "transcode failed (%d): %s\n", rc, error);
        return rc == 1 ? 3 : 1;
    }
    fprintf(stderr, "x265 build %d\n", rife60_x265_build_number());
    return 0;
}
#endif
