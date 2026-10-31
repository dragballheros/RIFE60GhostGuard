#!/usr/bin/env python3
"""Connect the pinned H3 runtime to on-device GGUF streaming."""
from pathlib import Path
import sys

def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"{path.name} is missing the expected GGUF patch site")
    if text.count(old) != 1:
        raise SystemExit(f"{path.name} patch site is not unique")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

def main() -> None:
    root = Path(sys.argv[1])
    weights = root / "h3_weights.c"
    gpu = root / "h3_gpu.m"
    core = root / "h3.c"
    inventory = root / "h3_safetensors.c"
    replace_once(
        weights,
        '#include "h3_weights.h"\n',
        '#include "h3_weights.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        weights,
        "        if (!safetensors_name(entry->d_name)) continue;\n",
        "        {\n"
        "            size_t name_length = strlen(entry->d_name);\n"
        "            int is_gguf = name_length > 5 &&\n"
        "                strcmp(entry->d_name + name_length - 5, \".gguf\") == 0;\n"
        "            if (!safetensors_name(entry->d_name) && !is_gguf) continue;\n"
        "        }\n",
    )
    replace_once(
        weights,
        "        if (!h3_st_read_header(paths[index], &store->headers[index], detail,\n"
        "                               sizeof(detail))) {\n",
        "        int gguf = h3_gguf_import_header(paths[index], &store->headers[index],\n"
        "                                     detail, sizeof(detail));\n"
        "        if (gguf < 0 || (gguf == 0 && !h3_st_read_header(\n"
        "                paths[index], &store->headers[index], detail, sizeof(detail)))) {\n",
    )
    replace_once(
        weights,
        "    if (tensor->dtype != dtype || tensor->ndim != ndim) {\n",
        "    if (!((dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16)) &&\n"
        "        (tensor->dtype != dtype || tensor->ndim != ndim)) {\n",
    )
    replace_once(
        weights,
        "    if (elements > SIZE_MAX) {\n",
        "    if (dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16) {\n"
        "        return h3_gguf_load_f16_as_f32(gpu, header->path, tensor->file_offset,\n"
        "                                       (size_t)elements, error, error_size);\n"
        "    }\n"
        "    if (elements > SIZE_MAX) {\n",
    )
    replace_once(
        gpu,
        '#include "h3_gpu.h"\n',
        '#include "h3_gpu.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        gpu,
        "            snprintf(error, error_size, \"%s file read range overflows\", label);\n"
        "        return 0;\n"
        "    }\n"
        "    int descriptor = open(path, O_RDONLY | O_CLOEXEC);\n",
        "            snprintf(error, error_size, \"%s file read range overflows\", label);\n"
        "        return 0;\n"
        "    }\n"
        "    if (h3_gguf_try_decode(path, file_offset, TENSOR(opaque).buffer.contents,\n"
        "                           elements, dtype == H3_GPU_BF16, error, error_size)) {\n"
        "        return !(error && error_size && error[0]);\n"
        "    }\n"
        "    int descriptor = open(path, O_RDONLY | O_CLOEXEC);\n",
    )
    replace_once(
        inventory,
        '#include "h3_safetensors.h"\n',
        '#include "h3_safetensors.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        inventory,
        '        if (entry->d_name[0] == \'.\' || !h3_has_suffix(entry->d_name, ".safetensors")) continue;\n',
        "        if (entry->d_name[0] == '.') continue;\n"
        "        int is_gguf = h3_has_suffix(entry->d_name, \".gguf\");\n"
        "        if (!is_gguf && !h3_has_suffix(entry->d_name, \".safetensors\")) continue;\n",
    )
    replace_once(
        inventory,
        "        int ok = h3_st_read_header(path, &header, error, error_size);\n",
        "        int ok = is_gguf ? h3_gguf_import_header(path, &header, error, error_size) == 1\n"
        "                         : h3_st_read_header(path, &header, error, error_size);\n",
    )
    replace_once(
        core,
        '#include "h3_tokenizer.h"\n',
        '#include "h3_tokenizer.h"\n#include "h3_gguf.h"\n',
    )
    replace_once(
        core,
        "h3_ctx *h3_load_dir(const char *model_dir) {\n",
        "h3_ctx *h3_load_dir(const char *model_dir) {\n"
        "    h3_gguf_stage_tier1(model_dir);\n",
    )
    print(f"Patched pinned H3 runtime in {root} for GGUF streaming")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_h3_gguf_runtime.py VENDOR_DIR")
    main()
