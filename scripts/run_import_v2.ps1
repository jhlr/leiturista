# Reimport completo do dataset Neoenergia (7.265 fotos) com a uniao de detectores
# (det original + det visor-ft da release modelos-1.2).
#
# Uso:  powershell -File scripts\run_import_v2.ps1
# Log:  data\neoenergia_amr_v2\import.log
#
# Roda desacoplado do terminal (Start-Process), entao sobrevive a queda da sessao.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$out = "data\neoenergia_amr_v2"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:LEITURISTA_DET2_ONNX = "models\pp_ocr_v5_mobile_det_visor_onnx\inference.onnx"
$env:LEITURISTA_DET2_THRESH = "0.2"
$env:LEITURISTA_DET2_BOX_THRESH = "0.2"

$log = Join-Path $out "import.log"
Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList "-m", "leiturista.cli", "import-distribuidora",
                  "--data-root", "data\FotosNeoenergia", "-o", $out `
    -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $out "import.err.log") `
    -WindowStyle Hidden

Write-Host "Import disparado. Progresso: (Get-ChildItem '$out\*.png').Count"
Write-Host "Log: $log"
