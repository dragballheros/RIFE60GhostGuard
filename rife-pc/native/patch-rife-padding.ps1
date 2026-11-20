# Sets RIFE padding from the model directory name.
# 4.6 stays at 32. 4.26 and 4.25 use 64. 4.25-lite uses 128.
# Called by build-windows.ps1 after the pinned rife-ncnn-vulkan checkout.
param([Parameter(Mandatory = $true)][string]$SrcRoot)

$cppPath = Join-Path $SrcRoot "src\rife.cpp"
$hPath = Join-Path $SrcRoot "src\rife.h"
if (-not (Test-Path $cppPath) -or -not (Test-Path $hPath)) {
    throw "rife sources were not checked out at $SrcRoot"
}
$cpp = [IO.File]::ReadAllText($cppPath)
$h = [IO.File]::ReadAllText($hPath)
if ($cpp.Contains("rife_padding")) {
    Write-Host "RIFE padding patch is already applied"
    return
}

$headerAnchor = "    bool rife_v4;`n};"
if (-not $h.Contains($headerAnchor)) {
    throw "rife.h no longer has the 'bool rife_v4' field this patch expects."
}
$h = $h.Replace($headerAnchor, "    bool rife_v4;`n    int rife_padding;`n};")

$ctorAnchor = "    rife_v4 = _rife_v4;`n}"
if (-not $cpp.Contains($ctorAnchor)) {
    throw "rife.cpp constructor no longer ends with rife_v4 = _rife_v4."
}
$cpp = $cpp.Replace($ctorAnchor, "    rife_v4 = _rife_v4;`n    rife_padding = 32;`n}")

$registerAnchor = "    fusionnet.register_custom_layer(`"rife.Warp`", Warp_layer_creator);`n"
if (-not $cpp.Contains($registerAnchor)) {
    throw "rife.cpp no longer registers rife.Warp where the padding patch expects."
}
$paddingBody = @"
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);
#if _WIN32
    rife_padding = 32;
    if (modeldir.find(L"4.25-lite") != std::wstring::npos) rife_padding = 128;
    else if (modeldir.find(L"4.26") != std::wstring::npos || modeldir.find(L"4.25") != std::wstring::npos) rife_padding = 64;
#else
    rife_padding = 32;
    if (modeldir.find("4.25-lite") != std::string::npos) rife_padding = 128;
    else if (modeldir.find("4.26") != std::string::npos || modeldir.find("4.25") != std::string::npos) rife_padding = 64;
#endif

"@
$cpp = $cpp.Replace($registerAnchor, $paddingBody.Replace("`r`n", "`n"))

$padAnchor = "    // pad to 32n`n    int w_padded = (w + 31) / 32 * 32;`n    int h_padded = (h + 31) / 32 * 32;"
$padCount = ([regex]::Matches($cpp, [regex]::Escape($padAnchor))).Count
if ($padCount -ne 4) {
    throw "Expected 4 pad-to-32 sites in rife.cpp, found $padCount."
}
$padNew = "    int align = rife_padding > 0 ? rife_padding : 32;`n    int w_padded = (w + align - 1) / align * align;`n    int h_padded = (h + align - 1) / align * align;"
$cpp = $cpp.Replace($padAnchor, $padNew)

$utf8 = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText($hPath, $h, $utf8)
[IO.File]::WriteAllText($cppPath, $cpp, $utf8)
Write-Host "Patched RIFE padding for 4.26 (64) and 4.6 (32)"
