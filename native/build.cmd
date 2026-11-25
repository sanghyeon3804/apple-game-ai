@echo off
rem Requires Visual Studio with C++ desktop tools
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /utf-8 /O2 /fp:fast /arch:AVX2 /MT /EHsc /std:c++17 /LD engine.cpp /Fe:engine.dll
