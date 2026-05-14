# Windows onedir build (Python 3.10+ and pip).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$constantsPath = Join-Path $Root "thermal_decoder\constants.py"
$constantsRaw = Get-Content -LiteralPath $constantsPath -Raw -Encoding UTF8
if ($constantsRaw -notmatch 'APP_VERSION\s*=\s*"([^"]+)"') {
    Write-Error "Could not read APP_VERSION from thermal_decoder/constants.py"
}
$AppVersion = $Matches[1]

python -m pip install -r requirements.txt -r requirements-build.txt

$BuilderRoot = Join-Path $Root "mainbuilder"
$DistRoot = Join-Path $BuilderRoot "dist"
$WorkRoot = Join-Path $BuilderRoot "build"
New-Item -ItemType Directory -Path $BuilderRoot -Force | Out-Null

python -m PyInstaller thermal_decoder_gui.spec --noconfirm `
    --distpath $DistRoot --workpath $WorkRoot

$DistDir = Join-Path $DistRoot "ThermalDecoder"
$Instr = Join-Path $Root "packaging\USER_GUIDE_RU.txt"
if (-not (Test-Path $DistDir)) {
    Write-Error "Build output missing: $DistDir"
}
Copy-Item -LiteralPath $Instr -Destination $DistDir -Force

$CertOut = Join-Path $DistDir "ThermalDecoder.cert"
python -m thermal_decoder.build_certificate --out $CertOut

$ZipName = "ThermalDecoder-$AppVersion-Windows.zip"
$ZipPath = Join-Path $DistRoot $ZipName
if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path $DistDir -DestinationPath $ZipPath -Force
Write-Host "OK dist: $DistDir"
Write-Host "OK zip: $ZipPath"
