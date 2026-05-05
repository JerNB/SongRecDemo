@echo off
REM ---------------------------------------------------------------------
REM  Start the local NetEase Cloud Music API service that the
REM  SongRecDemo product layer depends on.
REM
REM  The service is the third-party Node.js app at
REM    https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced
REM  cloned into  vendor/api-enhanced  (gitignored, not part of the
REM  research codebase).
REM
REM  Run from anywhere:
REM      scripts\start_netease_api.bat
REM
REM  The window stays open showing the API logs. Close it (or Ctrl+C)
REM  to stop the service. The service must be running for
REM  /api/song-search and /api/recommend to return live results.
REM ---------------------------------------------------------------------
setlocal
set "ROOT=%~dp0.."
set "API_DIR=%ROOT%\vendor\api-enhanced"

if not exist "%API_DIR%\app.js" (
    echo [ERROR] Could not find vendor\api-enhanced\app.js
    echo         Did you clone the upstream NetEase API service?
    echo         See SongRecDemo\README.md for setup steps.
    exit /b 1
)

if not exist "%API_DIR%\node_modules" (
    echo [INFO ] node_modules missing; running npm install once ...
    pushd "%API_DIR%" >nul
    call npm install --omit=dev --ignore-scripts ^
        --registry=https://registry.npmmirror.com ^
        --no-audit --no-fund
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd >nul
        exit /b 1
    )
    popd >nul
)

echo [INFO ] Starting NetEase Cloud Music API at http://localhost:3000
echo [INFO ] Leave this window open while using the demo.
echo.
pushd "%API_DIR%" >nul
node app.js
popd >nul

endlocal
