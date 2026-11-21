# Teach the pinned rife-ncnn-vulkan checkout a per-model pad alignment.
# rife-v4.6 stays at 32. rife-v4.26 and rife-v4.25 use 64. rife-v4.25-lite uses 128.
#
# Commit a7532fc still declares `bool rife_v4`. Git on the Windows runner checks
# that file out with CRLF, so matches are made after newlines are normalized.
# A missing anchor is not a failure when this change is already applied, or when
# upstream no longer hardcodes `(w + 31) / 32 * 32`.

param([Parameter(Mandatory = $true)][string]$SrcRoot)

$cppPath = Join-Path $SrcRoot "src\rife.cpp"
$hPath = Join-Path $SrcRoot "src\rife.h"
if (-not (Test-Path $cppPath) -or -not (Test-Path $hPath)) {
    throw "rife sources were not checked out at $SrcRoot"
}

function Get-NormalizedText([string]$Path, [ref]$UseCrLf) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    $text = [Text.Encoding]::UTF8.GetString($bytes)
    $UseCrLf.Value = $text.Contains("`r`n")
    return ($text -replace "`r`n", "`n" -replace "`r", "`n")
}

function Write-NormalizedText([string]$Path, [string]$Text, [bool]$UseCrLf) {
    if ($UseCrLf) {
        $Text = $Text -replace "(?<!`r)`n", "`r`n"
    }
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($Path, $Text, $utf8)
}

function Add-LineAfter([string]$Text, [string]$Pattern, [string]$AddedLine) {
    $regex = [regex]$Pattern
    $match = $regex.Match($Text)
    if (-not $match.Success) {
        return $null
    }
    $indent = $match.Groups[1].Value
    $replacement = $match.Value + "`n" + $indent + $AddedLine
    return $regex.Replace($Text, $replacement, 1)
}

$headerCrLf = $false
$cppCrLf = $false
$h = Get-NormalizedText $hPath ([ref]$headerCrLf)
$cpp = Get-NormalizedText $cppPath ([ref]$cppCrLf)

$padPattern = '(?m)^([ \t]*)int\s+w_padded\s*=\s*\(w\s*\+\s*31\)\s*/\s*32\s*\*\s*32\s*;\r?\n[ \t]*int\s+h_padded\s*=\s*\(h\s*\+\s*31\)\s*/\s*32\s*\*\s*32\s*;'
$padMatches = [regex]::Matches($cpp, $padPattern)
$needsPad = $padMatches.Count -gt 0
$alreadyUsesMember = $cpp -match 'rife_padding'

if (-not $needsPad -and -not $alreadyUsesMember) {
    Write-Host "RIFE padding patch: upstream has no hardcoded 32-pixel pad. Left rife.cpp unchanged."
    exit 0
}

$headerChanged = $false
if ($h -notmatch '(?m)^\s*int\s+rife_padding\s*;') {
    $updated = Add-LineAfter $h '(?m)^([ \t]*)bool\s+rife_v4\s*;' 'int rife_padding;'
    if (-not $updated) {
        $updated = Add-LineAfter $h '(?m)^([ \t]*)bool\s+rife_v2\s*;' 'int rife_padding;'
    }
    if (-not $updated) {
        $updated = Add-LineAfter $h '(?m)^([ \t]*)int\s+num_threads\s*;' 'int rife_padding;'
    }
    if ($updated) {
        $h = $updated
        $headerChanged = $true
        Write-Host "Added int rife_padding to rife.h"
    } elseif ($needsPad) {
        throw "Pad sites still use a hardcoded 32-pixel alignment, and rife.h has no field this patch can attach rife_padding to."
    } else {
        Write-Host "rife.h has no rife_padding member. Pad sites already avoid the hardcoded 32-pixel formula."
    }
} else {
    Write-Host "rife.h already declares rife_padding"
}

if ($cpp -notmatch '(?m)^\s*rife_padding\s*=') {
    $updated = Add-LineAfter $cpp '(?m)^([ \t]*)rife_v4\s*=\s*_rife_v4\s*;' 'rife_padding = 32;'
    if (-not $updated) {
        $updated = Add-LineAfter $cpp '(?m)^([ \t]*)rife_v2\s*=\s*_rife_v2\s*;' 'rife_padding = 32;'
    }
    if (-not $updated) {
        $updated = Add-LineAfter $cpp '(?m)^([ \t]*)num_threads\s*=\s*_num_threads\s*;' 'rife_padding = 32;'
    }
    if ($updated) {
        $cpp = $updated
        Write-Host "Initialized rife_padding in the RIFE constructor"
    } elseif ($needsPad) {
        throw "Could not find the RIFE constructor assignment that should initialize rife_padding."
    }
} else {
    Write-Host "RIFE constructor already sets rife_padding"
}

$modelRule = @'
#if _WIN32
    rife_padding = 32;
    if (modeldir.find(L"4.25-lite") != std::wstring::npos) rife_padding = 128;
    else if (modeldir.find(L"4.26") != std::wstring::npos || modeldir.find(L"4.25") != std::wstring::npos) rife_padding = 64;
#else
    rife_padding = 32;
    if (modeldir.find("4.25-lite") != std::string::npos) rife_padding = 128;
    else if (modeldir.find("4.26") != std::string::npos || modeldir.find("4.25") != std::string::npos) rife_padding = 64;
#endif
'@
$modelRule = $modelRule -replace "`r`n", "`n"

if ($cpp -notmatch 'modeldir\.find\([^\)]*4\.26') {
    $warp = 'fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);'
    $warpAt = $cpp.IndexOf($warp)
    if ($warpAt -ge 0) {
        $cpp = $cpp.Insert($warpAt + $warp.Length, "`n" + $modelRule)
        Write-Host "Set padding from the model directory inside RIFE::load"
    } else {
        $load = [regex]::Match($cpp, 'int\s+RIFE::load\s*\([^)]*\)\s*\{')
        if ($load.Success) {
            $cpp = $cpp.Insert($load.Index + $load.Length, "`n" + $modelRule)
            Write-Host "Set padding from the model directory at the start of RIFE::load"
        } elseif ($needsPad) {
            throw "RIFE::load was not found, so model-directory padding could not be set."
        }
    }
} else {
    Write-Host "RIFE::load already selects padding from the model directory"
}

if ($needsPad) {
    $cpp = [regex]::Replace($cpp, $padPattern, {
        param($m)
        $indent = $m.Groups[1].Value
        return @(
            "${indent}int align = rife_padding > 0 ? rife_padding : 32;",
            "${indent}int w_padded = (w + align - 1) / align * align;",
            "${indent}int h_padded = (h + align - 1) / align * align;"
        ) -join "`n"
    })
    Write-Host "Updated $($padMatches.Count) hardcoded 32-pixel pad sites"
} else {
    Write-Host "Pad sites already follow a configurable alignment"
}

if ($h -notmatch '(?m)^\s*int\s+rife_padding\s*;' -and $cpp -match 'rife_padding') {
    throw "rife.cpp now uses rife_padding, but rife.h has no rife_padding member."
}

Write-NormalizedText $hPath $h $headerCrLf
Write-NormalizedText $cppPath $cpp $cppCrLf
if (-not $headerChanged -and -not $needsPad) {
    Write-Host "RIFE padding patch: nothing to change"
}
Write-Host "RIFE padding patch finished (4.6 -> 32, 4.25/4.26 -> 64, 4.25-lite -> 128)"
