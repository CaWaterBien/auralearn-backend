@echo off
echo ================================
echo  TESTING AURABOT - SIMPLE VERSION
echo ================================

timeout /t 5 /nobreak

echo Testing basic connectivity...
powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000' -Method GET -TimeoutSec 5 | Out-Null; Write-Host 'SERVER RUNNING: YES' } catch { Write-Host 'SERVER RUNNING: NO' }"

echo.
echo Testing AuraBot endpoints...
powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/aurabot/health' -Method GET -TimeoutSec 5 | Out-Null; Write-Host 'HEALTH ENDPOINT: SUCCESS' } catch { Write-Host 'HEALTH ENDPOINT: FAIL' }"

powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/aurabot/session-status?session_id=test' -Method GET -TimeoutSec 5 | Out-Null; Write-Host 'SESSION ENDPOINT: SUCCESS' } catch { Write-Host 'SESSION ENDPOINT: FAIL' }"

echo.
echo ================================
echo If you see SUCCESS messages above, 
echo your AuraBot system is WORKING!
echo ================================
echo.
echo Next: Open frontend and test in activity page!
echo Frontend URL: http://localhost:3000
pause
