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

$build = Join-Path $Root "native\build"
# Applies to this configure and to the ncnn and glslang subdirectories.
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"
cmake -S (Join-Path $Root "native") -B $build "-DRIFE_SRC=$(Join-Path $srcRoot 'src')" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
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
