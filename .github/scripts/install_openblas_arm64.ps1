# Follow OpenBLAS's Windows ARM64 CI recipe, using its portable ARMV8 target:
# https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.33/.github/workflows/windows_arm64.yml
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$version = '0.3.33'
$archive = Join-Path $env:RUNNER_TEMP 'openblas-source.tar.gz'
$source = Join-Path $env:RUNNER_TEMP "OpenBLAS-$version"
$build = Join-Path $env:RUNNER_TEMP 'openblas-build'
$prefix = Join-Path $env:RUNNER_TEMP 'openblas'
$llvm = Join-Path $env:RUNNER_TEMP 'openblas-llvm'
$installer = Join-Path $env:RUNNER_TEMP 'LLVM-20.1.8-woa64.exe'

Invoke-WebRequest -OutFile $archive -Uri "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$version/OpenBLAS-$version.tar.gz"
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne '6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab') {
    throw 'OpenBLAS source checksum mismatch'
}
tar -xf $archive -C $env:RUNNER_TEMP

# Pin the C and Fortran toolchain used by the upstream recipe. Keep it separate
# from the runner's LLVM, which does not necessarily include flang-new.
Invoke-WebRequest -OutFile $installer -Uri 'https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/LLVM-20.1.8-woa64.exe'
if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne '7c4ac97eb2ae6b960ca5f9caf3ff6124c8d2a18cc07a7840a4d2ea15537bad8e') {
    throw 'LLVM installer checksum mismatch'
}
$install = Start-Process -FilePath $installer -ArgumentList '/S', "/D=$llvm" -Wait -PassThru
if ($install.ExitCode -ne 0) { throw "LLVM installation failed: $($install.ExitCode)" }

# Locate Visual Studio rather than depending on its version or edition.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
if (-not $vs) { throw 'Visual Studio ARM64 tools were not found' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsarm64.bat'
$environment = & cmd /d /c "`"`"$vcvars`" >nul && set`""
foreach ($line in $environment) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:PATH = "$llvm\bin;$env:PATH"

cmake -S $source -B $build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DTARGET=ARMV8 `
    -DDYNAMIC_ARCH=OFF `
    -DBINARY=64 `
    -DCMAKE_C_COMPILER=clang-cl `
    -DCMAKE_Fortran_COMPILER=flang-new `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_SYSTEM_PROCESSOR=arm64 `
    -DCMAKE_SYSTEM_NAME=Windows `
    "-DCMAKE_INSTALL_PREFIX=$prefix"
cmake --build $build --parallel 4
cmake --install $build

# Meson uses the CMake package; delvewheel bundles the DLL found on PATH.
"CMAKE_PREFIX_PATH=$prefix" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
"$prefix\bin" | Out-File -Append -Encoding utf8 $env:GITHUB_PATH
