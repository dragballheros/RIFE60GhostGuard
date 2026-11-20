; One setup executable. Installs under Program Files, adds PATH, Start Menu, uninstaller.

#define MyAppName "RIFE60 GhostGuard"
#define MyAppVersion "1.0.0"
#define MyAppExe "rife-pc.exe"

[Setup]
AppId={{7E3C1D26-B4F4-41EA-9C3D-0A1B2C3D4E5F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=RIFE60 GhostGuard
DefaultDirName={autopf}\RIFE60 GhostGuard
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=RIFE60GhostGuard-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
ChangesEnvironment=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExe}
LicenseFile=..\THIRD_PARTY.txt

[Files]
Source: "..\staging\rife-pc.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\staging\rife_capi.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\staging\ffmpeg.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\staging\ffprobe.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\staging\THIRD_PARTY.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\staging\models\rife-v4.6\*"; DestDir: "{app}\models\rife-v4.6"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\RIFE60 GhostGuard"; Filename: "{app}\{#MyAppExe}"; Parameters: "gui"; Comment: "Interpolate a video with RIFE on Vulkan"
Name: "{group}\Uninstall RIFE60 GhostGuard"; Filename: "{uninstallexe}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  AppDir: string;
  Needle: string;
  Prefix: string;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    if RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path) then
    begin
      Needle := ';' + AppDir;
      Prefix := AppDir + ';';
      if Pos(Uppercase(Prefix), Uppercase(Path)) = 1 then
        Path := Copy(Path, Length(Prefix) + 1, MaxInt)
      else if Pos(Uppercase(Needle), Uppercase(Path)) > 0 then
        StringChangeEx(Path, Needle, '', True);
      RegWriteExpandStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
    end;
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not FileExists(ExpandConstant('{sys}\vulkan-1.dll')) then
    MsgBox('vulkan-1.dll was not found in System32. Install the AMD Software: Adrenalin driver before running RIFE60 GhostGuard.', mbInformation, MB_OK);
end;
