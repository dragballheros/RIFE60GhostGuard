# RIFE60 GhostGuard — Windows build
# Produces installer\output\RIFE60GhostGuard-Setup.exe
# Requires an elevated PowerShell on Windows 11. Do not run this on the iOS tree's CI.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$RifeCommit = "a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7"
$VulkanVersion = "1.4.363.1"
$FfmpegZip = "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip"
$ModelZip = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"
$InnoUrl = "https://github.com/jrsoftware/issrc/releases/download/is-7_1_1/innosetup-7.1.1-x64.exe"

function Need([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing '$Name'. See rife-pc/README.md."
    }
}

function Get-File([string]$Url, [string]$Dest) {
    $headers = @{ "User-Agent" = "rife-pc-build" }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Dest -Headers $headers
            if ((Get-Item $Dest).Length -gt 0) { return }
            throw "Downloaded file is empty: $Url"
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds (15 * $attempt)
        }
    }
}

if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "Installing rustup"
    $rustup = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $rustup
    & $rustup -y --default-toolchain stable --profile minimal
    $env:Path = "$env:USERPROFILE\.cargo\bin;" + $env:Path
}
Need rustc
Need cargo
Need git
Need cmake

if (-not $env:VULKAN_SDK) {
    $sdk = "C:\VulkanSDK\$VulkanVersion"
    if (-not (Test-Path $sdk)) {
        Write-Host "Installing Vulkan SDK $VulkanVersion (compile-time only)"
        $sdkExe = Join-Path $env:TEMP "vulkansdk-windows-X64-$VulkanVersion.exe"
        Get-File "https://sdk.lunarg.com/sdk/download/$VulkanVersion/windows/vulkansdk-windows-X64-$VulkanVersion.exe?Human=true" $sdkExe
        $proc = Start-Process -FilePath $sdkExe -ArgumentList @("install","--accept-licenses","--confirm-command","--accept-messages","--root=$sdk") -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "Vulkan SDK installer exited $($proc.ExitCode)" }
        if (-not (Test-Path (Join-Path $sdk "Include\vulkan\vulkan.h"))) {
            throw "Vulkan SDK install did not produce $sdk\Include\vulkan\vulkan.h"
        }
    }
    $env:VULKAN_SDK = $sdk
}
$env:Path = "$env:VULKAN_SDK\Bin;" + $env:Path

$srcRoot = Join-Path $Root "third_party\rife-ncnn-vulkan"
if (-not (Test-Path (Join-Path $srcRoot "src\rife.cpp"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $srcRoot) | Out-Null
    git clone --filter=blob:none https://github.com/nihui/rife-ncnn-vulkan.git $srcRoot
}
git -C $srcRoot checkout $RifeCommit
if ($LASTEXITCODE -ne 0) {
    git -C $srcRoot fetch origin $RifeCommit
    git -C $srcRoot checkout FETCH_HEAD
}
git -C $srcRoot submodule update --init --recursive src/ncnn
# RIFE 4.26 needs 64-pixel padding. 4.25-lite needs 128. 4.6 stays at 32.
# The pinned runtime hardcodes 32. This edit keeps that commit building.
& (Join-Path $Root "native\patch-rife-padding.ps1") -SrcRoot $srcRoot

$build = Join-Path $Root "native\build"
# Applies to this configure and to the ncnn and glslang subdirectories.
# PowerShell splits an unquoted -DNAME=value argument on the equals sign when
# the value looks like a number (for example -DX=3.5). Keep every -D quoted.
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"
cmake -S (Join-Path $Root "native") -B $build "-DRIFE_SRC=$(Join-Path $srcRoot 'src')" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }

$dll = Get-ChildItem -Path $build -Recurse -Filter rife_capi.dll | Select-Object -First 1
$lib = Get-ChildItem -Path $build -Recurse -Filter rife_capi.lib | Select-Object -First 1
if (-not $dll -or -not $lib) { throw "rife_capi.dll/.lib were not produced" }
$dist = Join-Path $Root "native\dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Copy-Item $dll.FullName (Join-Path $dist "rife_capi.dll") -Force
Copy-Item $lib.FullName (Join-Path $dist "rife_capi.lib") -Force

$env:RIFE_CAPI_DIR = $dist
cargo build --release --manifest-path (Join-Path $Root "Cargo.toml") -p rife-pc --features ncnn,gui
if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }

$cache = Join-Path $Root "third_party\cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null

$ffmpegZipPath = Join-Path $cache "ffmpeg-gpl.zip"
if (-not (Test-Path $ffmpegZipPath)) {
    Write-Host "Downloading FFmpeg GPL (libx265)"
    Get-File $FfmpegZip $ffmpegZipPath
}
$ffmpegDir = Join-Path $cache "ffmpeg"
if (Test-Path $ffmpegDir) { Remove-Item -Recurse -Force $ffmpegDir }
Expand-Archive $ffmpegZipPath $ffmpegDir
$ffmpegExe = Get-ChildItem $ffmpegDir -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$ffprobeExe = Get-ChildItem $ffmpegDir -Recurse -Filter ffprobe.exe | Select-Object -First 1
if (-not $ffmpegExe -or -not $ffprobeExe) { throw "ffmpeg.exe/ffprobe.exe not in the GPL zip" }

$modelZipPath = Join-Path $cache "rife-ncnn-models.zip"
if (-not (Test-Path $modelZipPath)) {
    Write-Host "Downloading rife-v4.6 from the official ncnn-vulkan release"
    Get-File $ModelZip $modelZipPath
}
$modelDir = Join-Path $cache "models"
if (Test-Path $modelDir) { Remove-Item -Recurse -Force $modelDir }
Expand-Archive $modelZipPath $modelDir
$model = Get-ChildItem $modelDir -Recurse -Directory -Filter rife-v4.6 | Select-Object -First 1
if (-not $model) { throw "rife-v4.6 was not in the model zip" }
if (-not (Test-Path (Join-Path $model.FullName "flownet.param"))) { throw "flownet.param missing" }

$stage = Join-Path $Root "staging"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path (Join-Path $stage "models\rife-v4.6") | Out-Null
Copy-Item (Join-Path $Root "target\release\rife-pc.exe") (Join-Path $stage "rife-pc.exe")
Copy-Item (Join-Path $dist "rife_capi.dll") (Join-Path $stage "rife_capi.dll")
Copy-Item $ffmpegExe.FullName (Join-Path $stage "ffmpeg.exe")
Copy-Item $ffprobeExe.FullName (Join-Path $stage "ffprobe.exe")
Copy-Item (Join-Path $model.FullName "*") (Join-Path $stage "models\rife-v4.6") -Recurse
Copy-Item (Join-Path $Root "THIRD_PARTY.txt") (Join-Path $stage "THIRD_PARTY.txt")

$rife426 = Join-Path $stage "models\rife-v4.26"
New-Item -ItemType Directory -Force -Path $rife426 | Out-Null
Write-Host "Downloading rife-v4.26 flownet (ncnn, about 11 MB)"
Get-File "https://raw.githubusercontent.com/k4yt3x/video2x/master/models/rife/rife-v4.26/flownet.param" (Join-Path $rife426 "flownet.param")
Get-File "https://raw.githubusercontent.com/k4yt3x/video2x/master/models/rife/rife-v4.26/flownet.bin" (Join-Path $rife426 "flownet.bin")
if ((Get-Item (Join-Path $rife426 "flownet.bin")).Length -lt 1000000) {
    throw "rife-v4.26 flownet.bin is too small. The download was not the weight file."
}

$cuganZip = Join-Path $cache "realcugan-ncnn-vulkan-20220728-windows.zip"
if (-not (Test-Path $cuganZip)) {
    Write-Host "Downloading realcugan-ncnn-vulkan"
    Get-File "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-windows.zip" $cuganZip
}
$cuganDir = Join-Path $cache "realcugan"
if (Test-Path $cuganDir) { Remove-Item -Recurse -Force $cuganDir }
Expand-Archive $cuganZip $cuganDir
$cuganExe = Get-ChildItem $cuganDir -Recurse -Filter realcugan-ncnn-vulkan.exe | Select-Object -First 1
if (-not $cuganExe) { throw "realcugan-ncnn-vulkan.exe was not in the zip" }
Copy-Item $cuganExe.FullName (Join-Path $stage "realcugan-ncnn-vulkan.exe") -Force
Get-ChildItem $cuganExe.DirectoryName -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $stage $_.Name) -Force
}
$denoise = Get-ChildItem $cuganDir -Recurse -Filter "up2x-denoise3x.param" | Select-Object -First 1
if (-not $denoise) { throw "models-se up2x-denoise3x.param was not in the Real-CUGAN zip" }
$seStage = Join-Path $stage "models\models-se"
New-Item -ItemType Directory -Force -Path $seStage | Out-Null
Copy-Item (Join-Path $denoise.DirectoryName "up2x-denoise3x.param") $seStage
Copy-Item (Join-Path $denoise.DirectoryName "up2x-denoise3x.bin") $seStage

$ortVersion = "1.24.4"
$ortNupkg = Join-Path $cache "Microsoft.ML.OnnxRuntime.DirectML.$ortVersion.nupkg"
if (-not (Test-Path $ortNupkg)) {
    Write-Host "Downloading ONNX Runtime DirectML $ortVersion"
    Get-File "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/$ortVersion/microsoft.ml.onnxruntime.directml.$ortVersion.nupkg" $ortNupkg
}
$ortDir = Join-Path $cache "onnxruntime-directml"
if (Test-Path $ortDir) { Remove-Item -Recurse -Force $ortDir }
Copy-Item $ortNupkg (Join-Path $cache "onnxruntime-directml.zip") -Force
Expand-Archive (Join-Path $cache "onnxruntime-directml.zip") $ortDir
$ortInclude = Get-ChildItem $ortDir -Recurse -Filter onnxruntime_cxx_api.h | Select-Object -First 1
$ortLib = Get-ChildItem $ortDir -Recurse -Filter onnxruntime.lib | Select-Object -First 1
if (-not $ortInclude -or -not $ortLib) { throw "ONNX Runtime headers or import library were not in the DirectML package" }
$ortBuild = Join-Path $Root "native\ort-build"
# Quoted -D because PowerShell would otherwise split -DNAME=value.
cmake -S (Join-Path $Root "native\ort") -B $ortBuild "-DORT_INCLUDE=$($ortInclude.DirectoryName)" "-DORT_LIB=$($ortLib.FullName)" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime CMake configure failed with exit code $LASTEXITCODE" }
cmake --build $ortBuild --config Release
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime wrapper build failed with exit code $LASTEXITCODE" }
$ortCapi = Get-ChildItem $ortBuild -Recurse -Filter ort_capi.dll | Select-Object -First 1
if (-not $ortCapi) { throw "ort_capi.dll was not produced" }
Copy-Item $ortCapi.FullName (Join-Path $stage "ort_capi.dll") -Force
Get-ChildItem $ortDir -Recurse -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $stage $_.Name) -Force
}

$iscc = @(
    "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Write-Host "Installing Inno Setup 7.1.1"
    $inno = Join-Path $env:TEMP "innosetup-7.1.1-x64.exe"
    Get-File $InnoUrl $inno
    $proc = Start-Process -FilePath $inno -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART","/SP-" -Wait -PassThru
    if ($proc.ExitCode -ne 0) { throw "Inno Setup installer exited $($proc.ExitCode)" }
    $iscc = @(
        "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) { throw "ISCC.exe was not found after installing Inno Setup." }
}
& $iscc (Join-Path $Root "installer\rife-pc.iss")
Write-Host "Installer: $(Join-Path $Root 'installer\output\RIFE60GhostGuard-Setup.exe')"
