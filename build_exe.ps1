# exe 만들기: powershell -ExecutionPolicy Bypass -File build_exe.ps1
Set-Location $PSScriptRoot
python -m PyInstaller --noconfirm --clean --onefile --windowed --name AppleGameAI `
  --add-data "apple_ai/digit_templates.json;apple_ai" `
  --add-data "data/policy.json;data" `
  --add-binary "native/engine.dll;native" `
  app.py
exit $LASTEXITCODE
